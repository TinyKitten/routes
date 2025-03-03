use dotenv::dotenv;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use petgraph::{Graph, Undirected};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{MySql, Pool, mysql::MySqlPoolOptions};
use std::collections::BinaryHeap;
use std::{collections::HashMap, io};

#[allow(dead_code)]
#[derive(sqlx::FromRow, Debug)]
struct ConnectionRow {
    id: u32,
    station_cd1: u32,
    station_cd2: u32,
    distance: f64,
}

#[allow(dead_code)]
#[derive(sqlx::FromRow, Debug)]
struct StationRow {
    station_cd: u32,
    station_g_cd: u32,
    station_name: String,
    station_name_k: String,
    station_name_r: Option<String>,
    station_name_zh: Option<String>,
    station_name_ko: Option<String>,
    primary_station_number: Option<String>,
    secondary_station_number: Option<String>,
    extra_station_number: Option<String>,
    three_letter_code: Option<String>,
    line_cd: u32,
    pref_cd: u32,
    post: String,
    address: String,
    lon: f64,
    lat: f64,
    open_ymd: String,
    close_ymd: String,
    e_status: u32,
    e_sort: u32,
    distance: Option<f64>,
    line_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Content {
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Choice {
    delta: Content,
}

#[derive(Debug, Serialize, Deserialize)]
struct ResponseData {
    choices: Vec<Choice>,
}

fn dijkstra_with_path(
    graph: &Graph<i32, f64, Undirected>,
    start: NodeIndex,
) -> (
    HashMap<NodeIndex, f64>,
    HashMap<NodeIndex, Option<NodeIndex>>,
) {
    let mut dist_map = HashMap::new();
    let mut prev_map = HashMap::new();

    for node in graph.node_indices() {
        dist_map.insert(node, f64::INFINITY);
        prev_map.insert(node, None);
    }
    *dist_map.get_mut(&start).unwrap() = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0.0,
        node: start,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if cost > dist_map[&node] {
            continue;
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            let next_cost = cost + edge.weight();

            if next_cost < dist_map[&next] {
                *dist_map.get_mut(&next).unwrap() = next_cost;
                *prev_map.get_mut(&next).unwrap() = Some(node);
                heap.push(State {
                    cost: next_cost,
                    node: next,
                });
            }
        }
    }

    (dist_map, prev_map)
}

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    node: NodeIndex,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn reconstruct_path(
    prev_map: &HashMap<NodeIndex, Option<NodeIndex>>,
    start: NodeIndex,
    goal: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    let mut path = Vec::new();
    let mut current = goal;
    while let Some(&Some(prev)) = prev_map.get(&current) {
        path.push(current);
        current = prev;

        if current == start {
            path.push(start);
            path.reverse();
            return Some(path);
        }
    }
    None
}

async fn predict(
    from_place_name: String,
    destination_place_name: String,
    stop_stations: Vec<String>,
) -> Result<(), reqwest::Error> {
    let client = reqwest::Client::new();
    let mut res= client
        .post("http://localhost:1234/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(json!({
            "model": "gemma-2-9b-it",
            "messages": [{
                "role": "user",
                "content": format!("{}から{}まで鉄道で移動するのでおすすめの観光地を教えてください。ただし、下記の駅の付近限定でお願いします。\n{:?}", from_place_name,destination_place_name, stop_stations)
            }],
            "stream": true
        }).to_string())
        .send().await.unwrap().bytes_stream();

    while let Some(Ok(item)) = tokio_stream::StreamExt::next(&mut res).await {
        let s = String::from_utf8_lossy(&item).trim().replace("data: ", "");
        if let Ok(data) = serde_json::from_str::<ResponseData>(&s) {
            termimad::print_inline(&data.choices[0].delta.content);
        }
    }

    Ok(())
}

struct RouteFinder {
    pool: Pool<MySql>,
}

impl RouteFinder {
    async fn new(pool: Pool<MySql>) -> Self {
        RouteFinder { pool }
    }

    async fn get_all_nodes(&self) -> sqlx::Result<Vec<ConnectionRow>> {
        let mut conn = self.pool.acquire().await?;

        let rows = sqlx::query_as!(ConnectionRow, "SELECT * FROM connections")
            .fetch_all(&mut *conn)
            .await?;

        Ok(rows)
    }

    async fn find_station_by_name(&self, station_name: String) -> sqlx::Result<StationRow> {
        let mut conn: sqlx::pool::PoolConnection<MySql> = self.pool.acquire().await?;

        let station_row: StationRow = sqlx::query_as!(
            StationRow,
            "SELECT s.*,
            l.line_name,
            CAST(0.0 AS DOUBLE) AS distance
            FROM `stations` AS s
            JOIN `lines` as l ON l.line_cd = s.line_cd
            WHERE (
                    s.station_name = ?
                    OR s.station_name_r = ?
                    OR s.station_name_k = ?
                    OR s.station_name_zh = ?
                    OR s.station_name_ko = ?
                )
                AND s.e_status = 0
            LIMIT 1",
            &station_name,
            &station_name,
            &station_name,
            &station_name,
            &station_name,
        )
        .fetch_optional(&mut *conn)
        .await
        .expect("データベースでエラーが発生しました")
        .expect(&format!("や、{}駅のNASA🚀", station_name));

        Ok(station_row)
    }

    async fn find_edge_nodes(
        &self,
        from_place_name: String,
        destination_place_name: String,
    ) -> sqlx::Result<Vec<StationRow>> {
        let from_station_row = self.find_station_by_name(from_place_name).await.unwrap();
        let destination_station_row = self
            .find_station_by_name(destination_place_name)
            .await
            .unwrap();

        Ok(vec![from_station_row, destination_station_row])
    }

    async fn get_nodes_by_ids(&self, ids: &[u32]) -> sqlx::Result<Vec<StationRow>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let mut conn = self.pool.acquire().await?;

        let params = format!("?{}", ", ?".repeat(ids.len() - 1));

        let query_str = format!(
            "SELECT
            s.*,
            l.line_name,
            CAST(0.0 AS DOUBLE) AS distance
            FROM `stations` AS s
            JOIN `lines` AS l ON l.line_cd = s.line_cd AND l.e_status = 0
            WHERE
              s.station_cd IN ( {} )
              AND s.e_status = 0
            GROUP BY s.station_g_cd
            ORDER BY FIELD(s.station_cd, {})",
            params, params
        );

        let mut query = sqlx::query_as::<_, StationRow>(&query_str);
        for id in ids {
            query = query.bind(id);
        }
        for id in ids {
            query = query.bind(id);
        }

        let rows = query.fetch_all(&mut *conn).await?;

        Ok(rows)
    }

    async fn find_routes(
        &self,
        from_place_name: String,
        destination_place_name: String,
    ) -> sqlx::Result<Vec<StationRow>> {
        let conn_nodes = self.get_all_nodes().await?;

        let edge_nodes = self
            .find_edge_nodes(
                from_place_name.trim().to_string(),
                destination_place_name.trim().to_string(),
            )
            .await
            .expect("始発もしくは終着駅検索に失敗しました");

        let edges = conn_nodes.into_iter().map(|node| {
            let start = node.station_cd1;
            let goal = node.station_cd2;
            let weight = node.distance;
            (start, goal, weight)
        });

        let from_sta = edge_nodes.first().unwrap();
        let destination_sta = edge_nodes.last().unwrap();

        println!("\n始発駅: {}", from_sta.station_name);
        println!("終着駅: {}", destination_sta.station_name);

        let start_id = from_sta.station_cd;
        let goal_id = destination_sta.station_cd;

        let graph = UnGraph::<i32, f64>::from_edges(edges);
        let (dist_map, prev_map) = dijkstra_with_path(&graph, start_id.into());

        let mut stations: Vec<StationRow> = vec![];

        if let Some(path) = reconstruct_path(&prev_map, start_id.into(), goal_id.into()) {
            println!(
                "{} -> {} の最短距離: {}m",
                from_sta.station_name,
                destination_sta.station_name,
                dist_map[&goal_id.into()]
            );

            let node_ids: Vec<u32> = path.to_vec().iter().map(|x| x.index() as u32).collect();
            stations = self
                .get_nodes_by_ids(&node_ids)
                .await
                .expect("データベースでエラーが発生しました");
        } else {
            println!("や、経路のNASA🚀");
        }

        Ok(stations)
    }
}

#[tokio::main]
async fn main() {
    dotenv().ok();

    let mut from_place_name = String::new();
    let mut destination_place_name = String::new();

    println!("出発する地名を入力してください: ");
    io::stdin()
        .read_line(&mut from_place_name)
        .expect("I/Oエラーです");

    println!("行き先の地名を入力してください: ");
    io::stdin()
        .read_line(&mut destination_place_name)
        .expect("I/Oエラーです");

    let pool = MySqlPoolOptions::new()
        .max_connections(5)
        .connect(env!("DATABASE_URL"))
        .await
        .expect("MySQLに接続できませんでした");

    let finder = RouteFinder::new(pool).await;

    let stations = finder
        .find_routes(from_place_name.clone(), destination_place_name.clone())
        .await
        .expect("経路検索に失敗しました");

    let mut texts = String::new();

    for (index, station) in stations.iter().enumerate() {
        let stations_len = stations.len();
        texts += &format!(
            "{}({})",
            station.station_name,
            station.line_name.clone().unwrap()
        );

        if stations_len - 1 > index {
            texts += " -> ";
        }
    }

    println!("{}", texts);

    println!("\nAIのおすすめ情報:");
    predict(
        from_place_name,
        destination_place_name,
        stations.iter().map(|x| x.station_name.clone()).collect(),
    )
    .await
    .unwrap();
}
