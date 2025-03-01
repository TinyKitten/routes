use petgraph::visit::EdgeRef;
use std::collections::BinaryHeap;
use std::str::FromStr;
use std::{collections::HashMap, io};

use dotenv::dotenv;
use openai_api_rust::{
    Auth, Message, OpenAI, Role,
    chat::{ChatApi, ChatBody},
};
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::{Graph, Undirected};
use sqlx::{MySql, Pool, mysql::MySqlPoolOptions};

#[derive(sqlx::FromRow, Debug)]
struct ConnectionRow {
    id: u32,
    station_cd1: u32,
    station_cd2: u32,
    distance: f64,
}

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

fn predict(from_place_name: String, destination_place_name: String) -> io::Result<Vec<(f64, f64)>> {
    let auth = Auth::new("lm-studio");
    let openai = OpenAI::new(auth, "http://localhost:1234/v1/");

    let body = ChatBody {
        model: "gemma-2-9b-it".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: None,
        stream: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        messages: vec![Message {
            role: Role::User,
            content: format!("{} {}", from_place_name, destination_place_name).to_string(),
        }],
    };

    let rs = openai.chat_completion_create(&body);
    let choice = rs.unwrap().choices;
    let message = &choice[0].message.as_ref().unwrap();

    if message.content.contains("ERR") {
        panic!("AIが地名を見つけられませんでした");
    }

    let coordinates = message
        .content
        .split("\n")
        .collect::<Vec<&str>>()
        .iter()
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect::<Vec<String>>();

    if coordinates.len() < 2 {
        panic!("AIが正しく回答しませんでした");
    }

    Ok(coordinates
        .iter()
        .map(|x| {
            let lat_lon = x
                .split(" ")
                .collect::<Vec<&str>>()
                .iter()
                .map(|x| f64::from_str(x).unwrap())
                .collect::<Vec<f64>>();
            (lat_lon[0], lat_lon[1])
        })
        .collect())
}

struct RouteFinder {
    pool: Pool<MySql>,
    coordinates_pairs: Vec<(f64, f64)>,
}

impl RouteFinder {
    async fn new(pool: Pool<MySql>, coordinates_pairs: Vec<(f64, f64)>) -> Self {
        RouteFinder {
            pool,
            coordinates_pairs,
        }
    }

    async fn get_all_nodes(&self) -> sqlx::Result<Vec<ConnectionRow>> {
        let mut conn = self.pool.acquire().await?;

        let rows = sqlx::query_as!(ConnectionRow, "SELECT * FROM connections")
            .fetch_all(&mut *conn)
            .await?;

        Ok(rows)
    }

    async fn find_edge_nodes(&self) -> sqlx::Result<Vec<StationRow>> {
        let mut conn = self.pool.acquire().await?;

        let from_coords = self.coordinates_pairs.get(0).unwrap().clone();
        let destination_coords = self.coordinates_pairs.get(1).unwrap().clone();

        let rows = sqlx::query_as!(
            StationRow,
            "(SELECT
                s.*,
                l.line_name, 
                (
                  6371 * acos(
                    cos(
                      radians(s.lat)
                    ) * cos(
                      radians(?)
                    ) * cos(
                      radians(?) - radians(s.lon)
                    ) + sin(
                      radians(s.lat)
                    ) * sin(
                      radians(?)
                    )
                  )
                ) AS distance
              FROM `stations` AS s
              JOIN `lines` AS l ON l.line_cd = s.line_cd AND l.e_status = 0
              WHERE
                s.station_cd = s.station_g_cd
                AND s.e_status = 0
              ORDER BY 
                distance
              LIMIT 1)
              UNION
              (SELECT
                s.*,
                l.line_name, 
                (
                  6371 * acos(
                    cos(
                      radians(s.lat)
                    ) * cos(
                      radians(?)
                    ) * cos(
                      radians(?) - radians(s.lon)
                    ) + sin(
                      radians(s.lat)
                    ) * sin(
                      radians(?)
                    )
                  )
                ) AS distance
              FROM `stations` AS s
              JOIN `lines` AS l ON l.line_cd = s.line_cd AND l.e_status = 0
              WHERE
                s.station_cd = s.station_g_cd
                AND s.e_status = 0
              ORDER BY 
                distance
              LIMIT 1)",
            from_coords.0,
            from_coords.1,
            from_coords.0,
            destination_coords.0,
            destination_coords.1,
            destination_coords.0
        )
        .fetch_all(&mut *conn)
        .await?;

        Ok(rows)
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

    async fn find_routes(&self) -> sqlx::Result<Vec<StationRow>> {
        let conn_nodes = self.get_all_nodes().await?;

        let edge_nodes = self
            .find_edge_nodes()
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

        println!("");
        println!("AIが指定した始発駅: {}", from_sta.station_name);
        println!("AIが指定した終着駅: {}", destination_sta.station_name);
        println!("探索を開始しました...\n");

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

    let coordinates_pairs =
        predict(from_place_name, destination_place_name).expect("推論中にエラーが発生しました");

    let pool = MySqlPoolOptions::new()
        .max_connections(5)
        .connect(env!("DATABASE_URL"))
        .await
        .expect("MySQLに接続できませんでした");

    let finder = RouteFinder::new(pool, coordinates_pairs).await;

    let stations = finder.find_routes().await.expect("経路検索に失敗しました");
    for station in stations {
        println!(
            "{}({})",
            station.station_name,
            station.line_name.unwrap_or("???".to_string())
        );
    }
}
