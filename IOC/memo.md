 # ラプラス近似
 参考
 - http://triadsou.hatenablog.com/entry/20090217/1234865055

 # iterative LQR
これは概要
 https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/LQR.pdf

こっちのほうはがわかりやすいかも
https://katefvision.github.io/katefSlides/RECITATIONtrajectoryoptimization_katef.pdf

直感的に説明してくれているやつ

https://medium.com/@jonathan_hui/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750

これはsergey先生のやつ
http://rll.berkeley.edu/deeprlcoursesp17/docs/week_2_lecture_2_optimal_control.pdf

Iterative LQRはおそらく、終端状態が固定（目標値になる）と仮定して解く問題っぽい
これはMPCみたいにしないとダメだわ
結局やっていることとしては、ある時間までの有限時間最適化問題なんだけど
そのときに各タイムステップのモデルが変わっても大丈夫的な話をしている気がする
非線形かつ時変の問題を解法できるようなイメージ
