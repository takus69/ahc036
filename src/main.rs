use proconio::input;
use std::collections::{HashMap, VecDeque};

struct Solver {
    n: usize,
    m: usize,
    t: usize,
    la: usize,
    lb: usize,
    g: HashMap<usize, Vec<usize>>,
    t_list: Vec<usize>,
    xy: Vec<(usize, usize)>,
    a: Vec<usize>,
    b: Vec<usize>,
    ans: Vec<String>,
    score: usize,
}

impl Solver {
    fn new(n: usize, m: usize, t: usize, la: usize, lb: usize, uv: Vec<(usize, usize)>, t_list: Vec<usize>, xy: Vec<(usize, usize)>) -> Solver {
        let mut g = HashMap::new();
        for v in 0..n {
            g.insert(v, vec![]);
        }
        for (u, v) in uv.iter() {
            let e = g.get_mut(u).unwrap();
            e.push(*v);
            let e = g.get_mut(v).unwrap();
            e.push(*u);
        }
        let mut a = vec![0; la];
        for i in 0..la {
            if i < n {
                a[i] = i;
            }
        }
        let b = vec![usize::MAX; n];
        let ans: Vec<String> = Vec::new();
        let score = 0;

        Solver { n, m, t, la, lb, g, t_list, xy, a, b, ans, score }
    }

    fn dfs(&self, from: &usize, to: &usize, path: &mut Vec<usize>, visited: &mut Vec<bool>) -> bool {
        if visited[*from] { return false; }
        path.push(*from);
        if from == to { return true; }
        visited[*from] = true;
        for next in self.g.get(from).unwrap().iter() {
            if self.dfs(next, to, path, visited) {
                return true;
            }
        }

        path.pop();
        false
    }

    fn bfs(&self, from: usize, to: usize) -> Vec<usize> {
        let mut prev: Vec<usize> = vec![usize::MAX; self.n];
        let mut que: VecDeque<usize> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        que.push_front(from);
        visited[from] = true;
        while !que.is_empty() {
            let u = que.pop_back().unwrap();
            if u == to { break; }
            for v in self.g.get(&u).unwrap().iter() {
                if visited[*v] { continue; }
                que.push_front(*v);
                visited[*v] = true;
                prev[*v] = u;
            }
        }

        // 経路復元
        let mut path: Vec<usize> = Vec::new();
        path.push(to);
        let mut next = to;
        while next != from {
            next = prev[next];
            path.push(next);
        }
        path.reverse();

        path
    }

    fn solve(&mut self) {
        let mut from = 0;
        for to in self.t_list.iter() {
            // let mut visited = vec![false; self.n];
            // let mut path: Vec<usize> = Vec::new();

            // self.dfs(from, to, &mut path, &mut visited);  // 最短経路を探す
            println!("# from: {}, to: {}", from, to);
            let path = self.bfs(from, *to);
            for p in path[1..].iter() {
                self.ans.push(format!("s {} {} {}", 1, p, 0));
                self.score += 1;
                self.ans.push(format!("m {}", p));
            }
            from = *to;
        }

    }
    
    fn ans(self) {
        println!("{}", self.a.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        for a in self.ans.iter() {
            println!("{}", a);
        }
        eprintln!("{{ \"score\": {} }}", self.score);
    }
}

fn main() {
    input! {
        n: usize,
        m: usize,
        t: usize,
        la: usize,
        lb: usize,
        uv: [(usize, usize); m],
        t_list: [usize; t],
        xy: [(usize, usize); n],
    }

    let mut solver = Solver::new(n, m, t, la, lb, uv, t_list, xy);
    solver.solve();
    solver.ans();
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_new() {
        let n = 7;
        let m = 9;
        let t = 3;
        let la = 7;
        let lb = 4;
        let uv = vec![
            (0 ,1), (0 ,2), (0 ,3), (1 ,2), (2 ,3),
            (3 ,4), (4 ,5), (5 ,6), (6 ,0),
        ];
        let t_list = vec![4, 1, 5];
        let xy = vec![
            (100, 0), (200, 0), (200, 100), (100, 100),
            (0, 200), (0, 100), (0, 0),
        ];
        let solver = Solver::new(n, m, t, la, lb, uv, t_list, xy);
        assert_eq!(solver.g[&0], [1, 2, 3, 6]);
        assert_eq!(solver.g[&1], [0, 2]);
        assert_eq!(solver.g[&2], [0, 1, 3]);
        assert_eq!(solver.g[&3], [0, 2, 4]);
        assert_eq!(solver.g[&4], [3, 5]);
        assert_eq!(solver.g[&5], [4, 6]);
        assert_eq!(solver.g[&6], [5, 0]);
    }
}
