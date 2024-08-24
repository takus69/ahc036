use proconio::input;
use std::collections::{HashMap, VecDeque};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

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
    lt: usize,
    rng: StdRng,
    match_rate: f64,
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
        let b = vec![usize::MAX; lb];
        let ans: Vec<String> = Vec::new();
        let score = 0;
        let lt = 0;

        let seed: [u8; 32] = [0; 32];
        let rng = StdRng::from_seed(seed);
        let match_rate: f64 = 0.0;

        Solver { n, m, t, la, lb, g, t_list, xy, a, b, ans, score, lt, rng, match_rate }
    }

    fn optimize_a(&mut self) {
        let mut a: Vec<usize> = Vec::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        let mut next = 0;
        let mut now = next;
        while a.len() < self.n {
            now = next;
            a.push(now);
            visited[now] = true;
            for v in self.g.get(&now).unwrap().iter() {
                if !visited[*v] {
                    next = *v;
                    break;
                }
            }
            if next == now {
                for (i, vi) in visited.iter().enumerate() {
                    if !vi {
                        next = i;
                    }
                }
            }
        }
        let mut prev = now;
        while a.len() < self.la {
            now = next;
            a.push(now);
            let list = self.g.get(&now).unwrap();
            if list.len() > 1 {
                let i = self.rng.gen_range(0..list.len());
                next = list[i];
            }
            if list.len() == 1 || next == prev {
                next = self.rng.gen_range(0..self.n);
            }
            prev = now;
        }

        // 配列Aとpathの一致率が一番よくなるように最適化する
        let mut from = 0;
        let t_list = self.t_list.clone();
        let mut path: Vec<usize> = Vec::new();
        for to in t_list.iter() {
            let pathes = self.bfs(from, *to, 1);
            path.extend(pathes[0].clone());
            from = *to;
        }
        let mut match_cnt = 0;
        let mut all_cnt = 0;
        for i in 0..path.len() {
            let target = path[i..(i+self.lb).min(path.len())].to_vec();
            let (min_i, max_i) = self.r#match(&target, &a);
            match_cnt += max_i - min_i + 1;
            all_cnt += target.len();
        }
        self.match_rate = match_cnt as f64 / all_cnt as f64;

        println!("# a: {:?}", &a);
        self.a = a;
    }

    fn bfs(&self, from: usize, to: usize, max_cnt: usize) -> Vec<Vec<usize>> {
        let mut pathes: Vec<Vec<usize>> = Vec::new();
        let mut prev: Vec<usize> = vec![usize::MAX; self.n];
        let mut que: VecDeque<usize> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        que.push_front(from);
        visited[from] = true;
        while !que.is_empty() {
            let u = que.pop_back().unwrap();
            if u == to {
                let path = reconstruct(from, to, &prev);
                pathes.push(path);
                if pathes.len() >= max_cnt {
                    break;
                } else {
                    visited[to] = false;
                    continue;
                }
            }
            for v in self.g.get(&u).unwrap().iter() {
                if visited[*v] { continue; }
                que.push_front(*v);
                visited[*v] = true;
                prev[*v] = u;
            }
        }

        // 経路復元
        fn reconstruct(from: usize, to: usize, prev: &Vec<usize>) -> Vec<usize> {
            let mut path: Vec<usize> = Vec::new();
            path.push(to);
            let mut next = to;
            while next != from {
                next = prev[next];
                path.push(next);
            }
            path.pop();
            path.reverse();

            path
        }

        pathes
    }

    fn solve(&mut self) {
        let mut from = 0;
        let t_list = self.t_list.clone();
        self.optimize_a();
        for to in t_list.iter() {
            println!("# from: {}, to: {}", from, to);
            
            // 複数の経路から最適な経路を選択する
            let pathes = self.bfs(from, *to, 10);
            self.lt += pathes[0].len()-1;  // 理想的な値は一番短い経路から取得
            let mut opt_path_i = 0;
            let mut opt_switch_cnt = usize::MAX;
            for (i, path) in pathes.iter().enumerate() {
                let mut switch_cnt = 0;
                let mut b = self.b.clone();
                for (i, p) in path.iter().enumerate() {
                    let target = path[i..path.len().min(i+self.lb)].to_vec();
                    if !self.can_move(p, &b) {
                        let (l, s_a, s_b) = self.switch(&target);
                        b[s_b..(s_b+l)].clone_from_slice(&self.a[s_a..(s_a+l)]);
                        switch_cnt += 1;
                    }
                }
                if opt_switch_cnt > switch_cnt {
                    opt_switch_cnt = switch_cnt;
                    opt_path_i = i;
                }
            }
            // 最適な経路で実施
            let path = &pathes[opt_path_i];
            for (i, p) in path.iter().enumerate() {
                let target = path[i..path.len().min(i+self.lb)].to_vec();
                if !self.can_move(p, &self.b) {
                    let (l, s_a, s_b) = self.switch(&target);
                    self.switch_op(l, s_a, s_b);
                    println!("# target: {:?}, b: {:?}", target, self.b);
                }
                self.r#move(p);
            }
            from = *to;
        }
    }

    fn r#match(&self, target: &[usize], a: &[usize]) -> (usize, usize) {
        let mut min_i = usize::MAX;
        let mut max_i = 0;
        let mut candidates = vec![(min_i, max_i)];
        for t in target.iter() {
            let indices: Vec<usize> = a.iter().enumerate().filter(|(_, &x)| x == *t).map(|(i, _)| i).collect();
            let mut tmp: Vec<(usize, usize)> = Vec::new();
            for (min_i, max_i) in candidates.iter() {
                for index in indices.iter() {
                    let dist = max_i.max(index) - min_i.min(index) + 1;
                    if dist > self.lb {
                        continue;
                    } else {
                        tmp.push((*min_i.min(index), *max_i.max(index)));
                    }
                }
            }
            if tmp.is_empty() { break; } else { candidates = tmp; }
        }
        (min_i, max_i) = candidates[0];
        (min_i, max_i)
    }

    fn switch(&mut self, target: &[usize]) -> (usize, usize, usize) {
        let (min_i, max_i) = self.r#match(target, &self.a);
        let l = max_i - min_i + 1;
        let s_a = min_i;
        let s_b = 0;
        (l, s_a, s_b)
    }

    fn switch_op(&mut self, l: usize, s_a: usize, s_b: usize) {
        println!("# s {} {} {}", l, s_a, s_b);
        self.b[s_b..(s_b+l)].clone_from_slice(&self.a[s_a..(s_a+l)]); 
        self.ans.push(format!("s {} {} {}", l, s_a, s_b));
        self.score += 1;
    }

    fn can_move(&self, p: &usize, b: &[usize]) -> bool {
        b.contains(p)
    }
    
    fn r#move(&mut self, p: &usize) {
        println!("# m {}", p);
        self.ans.push(format!("m {}", p));
    }
    
    fn ans(self) {
        println!("{}", self.a.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        for a in self.ans.iter() {
            println!("{}", a);
        }
        eprintln!("{{ \"score\": {}, \"lt_lb\": {}, \"match_rate\": {} }}", self.score, (self.lt-1)/self.lb+1, self.match_rate);
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

    fn setup() -> Solver {
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
        Solver::new(n, m, t, la, lb, uv, t_list, xy)
    }
    
    #[test]
    fn test_switch() {
        let mut solver = setup();
        assert_eq!(solver.a, [0, 1, 2, 3, 4, 5, 6]);
        let (l, s_a, s_b) = solver.switch(&[3, 0, 2, 4]);
        solver.switch_op(l, s_a, s_b);
        assert_eq!(solver.b, [0, 1, 2, 3]);
    }

    fn test_bfs() {
        let mut solver = setup();
        let pathes = solver.bfs(0, 1, 2);
        assert_eq!(pathes.len(), 2);
        assert_eq!(pathes[0], [1, 0, 3]);
        assert_eq!(pathes[1], [1, 2, 3]);
        let pathes = solver.bfs(0, 1, 3);
        assert_eq!(pathes.len(), 2);
    }
}
