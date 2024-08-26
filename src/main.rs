use proconio::input;
use std::collections::{HashMap, VecDeque, BinaryHeap, HashSet};
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
    switch_match_cnt: Vec<usize>,
    all_cnt: usize,
    a_opt: AOptimizer,
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
        let switch_match_cnt: Vec<usize> = Vec::new();
        let all_cnt = 0;
        let a_opt = AOptimizer::new(n, la, lb);

        Solver { n, m, t, la, lb, g, t_list, xy, a, b, ans, score, lt, rng, match_rate, switch_match_cnt, all_cnt, a_opt }
    }

    fn get_path(&self) -> Vec<usize> {
        // 最適な経路を取得する
        let mut from = 0;
        let t_list = self.t_list.clone();
        let mut path: Vec<usize> = Vec::new();
        for to in t_list.iter() {
            let pathes = self.bfs(from, *to, 1);
            assert_ne!(pathes[0][0], from);
            assert_eq!(pathes[0][pathes[0].len()-1], *to);
            path.extend(pathes[0].to_vec());
            from = *to;
        }

        path
    }

    fn optimize_a(&mut self) {
        // 配列Aとpathの一致率が一番よくなるように最適化する
        let path = self.get_path();  // 最適な経路を取得

        // pathに対するLB範囲の出現数を算出
        let mut freq: Vec<Vec<usize>> = vec![vec![0; self.n]; self.n];
        for i in 0..path.len() {
            let t1 = path[i];
            for j in 1..2 {  // self.lb {
                if i+j == path.len() { break; }
                let t2 = path[i+j];
                freq[t1][t2] += 1;
                freq[t2][t1] += 1;
            }
        }

       // 優先度付きキューで、出現数が多い都市を順に配列Aに追加していく
        let mut heaps: Vec<BinaryHeap<(usize, usize)>> = vec![BinaryHeap::new(); self.n];
        for i in 0..self.n {
            for (j, cnt) in freq[i].iter().enumerate() {
                if i == j || cnt == &0 { continue; }
                heaps[i].push((*cnt, j));
            }
        }

        // 都市の道に沿って配列Aを設定
        let mut a: Vec<usize> = Vec::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        let mut next = 0;
        let mut now = next;
        while a.len() < self.n {
            now = next;
            a.push(now);
            visited[now] = true;
            let f = &freq[now];
            let mut max_freq = 0;
            let mut opt_v = 0;
            for v in self.g.get(&now).unwrap().iter() {
                if !visited[*v] && f[*v] > max_freq {
                    max_freq = f[*v];
                    opt_v = *v;
                }
            }
            if max_freq > 0 {
                next = opt_v;
            }
            if next == now {
                for (i, vi) in visited.iter().enumerate() {
                    if !vi {
                        next = i;
                    }
                }
            }
        }
        // 入らなかったインデックスを入れる
        let mut cnt = 0;
        for (i, v) in visited.iter().enumerate() {
            if !v {
                a.push(i);
                cnt += 1;
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

        // 配列Aとpathの一致率を算出
        self.a_opt.init(&path, &a);
        self.match_rate = self.a_opt.match_rate();
        println!("# match_rate: {}", self.match_rate);

        // 山登りで最適化

        // 最適化後の配列Aとpathの一致率を算出
        self.match_rate = self.a_opt.match_rate();
        println!("# match_rate: {}", self.match_rate);

        println!("# path: {}, lb: {}", path.len(), self.lb);

        assert_eq!(a.len(), self.la, "Length of a is not equal LA. a: {}, LA: {}", a.len(), self.la);
        self.a = a;
    }

    fn bfs(&self, from: usize, to: usize, max_cnt: usize) -> Vec<Vec<usize>> {
        let mut pathes: Vec<Vec<usize>> = Vec::new();
        let mut prev: Vec<usize> = vec![usize::MAX; self.n];
        let mut que: VecDeque<usize> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        que.push_front(from);
        while !que.is_empty() {
            let u = que.pop_back().unwrap();
            if u == to {
                let path = reconstruct(from, to, &prev);
                pathes.push(path);
                if pathes.len() >= max_cnt {
                    break;
                } else {
                    visited[to] = false;
                    prev[to] = usize::MAX;
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
            path.pop();  // fromを経路から削除
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
            self.lt += pathes[0].len();  // 理想的な値は一番短い経路から取得
            let mut opt_path_i = 0;
            let mut opt_switch_cnt = usize::MAX;
            for (j, path) in pathes.iter().enumerate() {
                let mut switch_cnt = 0;
                let mut b = self.b.clone();
                for (i, p) in path[1..].iter().enumerate() {
                    let target = path[i..path.len().min(i+self.lb)].to_vec();
                    if !self.can_move(p, &b) {
                        let (l, s_a, s_b, _) = self.switch(&target);
                        b[s_b..(s_b+l)].clone_from_slice(&self.a[s_a..(s_a+l)]);
                        switch_cnt += 1;
                    }
                }
                if opt_switch_cnt > switch_cnt {
                    opt_switch_cnt = switch_cnt;
                    opt_path_i = j;
                }
            }
            // 最適な経路で実施
            let path = &pathes[opt_path_i];
            println!("# path: {:?}", path);
            for (i, p) in path.iter().enumerate() {
                let target = path[i..path.len().min(i+self.lb)].to_vec();
                if !self.can_move(p, &self.b) {
                    let (l, s_a, s_b, match_cnt) = self.switch(&target);
                    self.switch_op(l, s_a, s_b);
                    self.switch_match_cnt.push(match_cnt);
                    self.all_cnt += self.lb;
                    println!("# target: {:?}, b: {:?}", target, self.b);
                }
                self.r#move(p);
            }
            from = *to;
        }
    }

    fn switch(&mut self, target: &[usize]) -> (usize, usize, usize, usize) {
        let (min_i, max_i, match_cnt) = self.a_opt.r#match(target);
        let l = max_i - min_i + 1;
        let s_a = min_i;
        let s_b = 0;
        (l, s_a, s_b, match_cnt)
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
        let match_rate  =self.switch_match_cnt.iter().sum::<usize>() as f64 / self.all_cnt as f64;
        eprintln!("{{ \"M\": {}, \"LA\": {}, \"LB\": {}, \"score\": {}, \"lt_lb\": {}, \"pred_match_rate\": {}, \"actual_match_rate\": {} }}", self.m, self.la, self.lb, self.score, (self.lt-1)/self.lb+1, self.match_rate, match_rate);
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

struct AOptimizer {
    n: usize,
    la: usize,
    lb: usize,
    path: Vec<usize>,
    a: Vec<usize>,
    path_freq: Vec<HashMap<usize, usize>>,
    eval: f64,
}

impl AOptimizer {
    fn new(n: usize, la: usize, lb: usize) -> AOptimizer {
        let path: Vec<usize> = Vec::new();
        let a: Vec<usize> = Vec::new();
        let path_freq: Vec<HashMap<usize, usize>> = Vec::new();
        let eval = 0.0;
        AOptimizer { n, la, lb, path, a, path_freq, eval }
    }
    
    fn init(&mut self, path: &Vec<usize>, a: &Vec<usize>) {
        self.path = path.clone();
        self.a = a.clone();
        self.a.extend(self.a[0..(2*self.lb)].to_vec());  // 配列Aをそのままで1週できるように拡張する

        let (path_freq, all_path_freq) = self.get_path_freq();  // pathにおける各都市の隣接都市情報を取得

        // 配列aのLB範囲に存在する経路の隣接都市の個数
        let mut match_cnt: Vec<HashMap<usize, usize>> = vec![HashMap::new(); self.n];
        let mut eval_freq: Vec<usize> = vec![0; self.n];
        for t in 0..self.n {
            let indices: Vec<usize> = self.a[0..self.la].iter().enumerate().filter(|(_, &x)| x == t).map(|(i, _)| i).collect();
            for i in indices.iter() {
                let from = if *i < self.lb { self.n+i-self.lb+1 } else { i-self.lb+1 };
                let to = from + 2*self.lb - 1;
                for j in from..to {
                    let aj = self.a[j];
                    if aj == t { continue; }
                    if path_freq[t].contains_key(&aj) {
                        let e = match_cnt[t].entry(aj).or_insert(0);
                        *e += 1;
                        if *e == 1 {
                            eval_freq[t] += path_freq[t].get(&aj).unwrap();
                        }
                    }
                }
            }
        }

        let path_set: HashSet<usize> = path.clone().into_iter().collect();

    }

    fn get_path_freq(&self) -> (Vec<HashMap<usize, usize>>, Vec<usize>) {
        let mut freq: Vec<HashMap<usize, usize>> = vec![HashMap::new(); self.n];
        for i in 0..(self.path.len()-1) {
            let t1 = self.path[i];
            let t2 = self.path[i+1];
            let e = freq[t1].entry(t2).or_insert(0);  // 隣接する都市の出現数
            *e += 1;
            let e = freq[t2].entry(t1).or_insert(0);  // 隣接する都市の出現数
            *e += 1;
        }

        // pathに存在しない都市はall_freqの初期値1, 存在する都市は初期値0
        let path_set: HashSet<usize> = self.path.clone().into_iter().collect();
        let mut all_freq: Vec<usize> = vec![1; self.n];
        for p in path_set.iter() {
            all_freq[*p] = 0;
        }
        for i in 0..self.n {
            for (_, cnt) in freq[i].iter() {
                all_freq[i] += cnt;
            }
        }

        (freq, all_freq)
    }

    fn r#match(&self, target: &[usize]) -> (usize, usize, usize) {
        let mut min_i = usize::MAX;
        let mut max_i = 0;
        let mut match_cnt = 0;
        let mut candidates = vec![(min_i, max_i)];
        for t in target.iter() {
            let indices: Vec<usize> = self.a[0..self.la].iter().enumerate().filter(|(_, &x)| x == *t).map(|(i, _)| i).collect();
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
            if tmp.is_empty() { break; } else {
                candidates = tmp;
                match_cnt += 1;
            }
        }
        (min_i, max_i) = candidates[0];
        (min_i, max_i, match_cnt)
    }

    fn match_rate(&self) -> f64 {
        let mut match_cnt = 0;
        let mut all_cnt = 0;
        for i in 0..self.path.len() {
            let target = self.path[i..(i+self.lb).min(self.path.len())].to_vec();
            let (_, _, cnt) = self.r#match(&target);
            match_cnt += cnt;
            all_cnt += target.len();
        }
        match_cnt as f64 / all_cnt as f64
    }

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
        solver.a_opt.a = vec![0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0];
        let (l, s_a, s_b, match_cnt) = solver.switch(&[3, 0, 2, 4]);
        solver.switch_op(l, s_a, s_b);
        assert_eq!(solver.b, [0, 1, 2, 3]);
        assert_eq!(match_cnt, 3);
    }

    #[test]
    fn test_bfs() {
        let mut solver = setup();
        let pathes = solver.bfs(1, 3, 2);
        assert_eq!(pathes.len(), 2);
        assert_eq!(pathes[0], [0, 3]);
        assert_eq!(pathes[1], [0, 6, 5, 4, 3]);
        let pathes = solver.bfs(1, 3, 3);
        assert_eq!(pathes.len(), 2);
    }

    #[test]
    fn test_a_opt() {
        let (n, la, lb) = (6, 6, 3);
        let path = vec![0, 1, 3, 1, 2, 4, 5, 1];
        let a = vec![0, 1, 2, 3, 4, 5];
        let mut a_opt = AOptimizer::new(n, la, lb);
        a_opt.init(&path, &a);
        let match_rate = a_opt.match_rate();
        assert_eq!(match_rate, 16.0/21.0);
    }
}
