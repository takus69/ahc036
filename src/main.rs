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
    lt: usize,
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

        Solver { n, m, t, la, lb, g, t_list, xy, a, b, ans, score, lt }
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
        path.pop();
        path.reverse();

        path
    }

    fn solve(&mut self) {
        let mut from = 0;
        let t_list = self.t_list.clone();
        for to in t_list.iter() {
            println!("# from: {}, to: {}", from, to);
            let path = self.bfs(from, *to);
            self.lt += path.len()-1;
            for (i, p) in path.iter().enumerate() {
                let target = path[i..path.len().min(i+self.lb)].to_vec();
                if !self.can_move(p) {
                    self.switch(target);
                }
                self.r#move(p);
            }
            from = *to;
        }

    }

    fn switch(&mut self, target: Vec<usize>) {
        let mut min_i = usize::MAX;
        let mut max_i = 0;
        for t in target.iter() {
            let index = self.a.iter().position(|&x| x == *t).unwrap();
            let dist = max_i.max(index) - min_i.min(index) + 1;
            if dist > self.lb {
                break;
            } else {
                min_i = min_i.min(index);
                max_i = max_i.max(index);
            }
        }
        let l = max_i - min_i + 1;
        let s_a = min_i;
        let s_b = 0;
        println!("# s {} {} {}", l, s_a, s_b);
        self.b[s_b..(s_b+l)].clone_from_slice(&self.a[s_a..(s_a+l)]); 
        self.ans.push(format!("s {} {} {}", l, s_a, s_b));
        self.score += 1;

    }

    fn can_move(&self, p: &usize) -> bool {
        self.b.contains(p)
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
        eprintln!("{{ \"score\": {}, \"lt_lb\": {} }}", self.score, (self.lt-1)/self.lb+1);
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
        solver.switch(vec![3, 0, 2, 4]);
        assert_eq!(solver.b, [0, 1, 2, 3]);
    }
}
