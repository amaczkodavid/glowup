# Szuperoptimalizáló Szintézis-verem

Egy teljes körű rendszer, amely **szintetizálja, verifikálja és futtatja** az ultra-optimalizált kódot, egyesítve a sztokasztikus szuperoptimalizálást, a megerősítéses tanulású assembly-szintézist, az egyenlőségi telítést ILP-kinyeréssel, a formális verifikációt, valamint egy rendkívül alacsony szintű Zig futtatókörnyezetet GPU-kollektívákkal és Futhark-integrációkkal.

```
optimizer (TypeScript, src/lib/superopt)
  isa.ts          újracélozható ISA-abstrakció (x86-64, AArch64, bővíthető)
  machine.ts      CPU/memóriamodell, hibakezelés, függőségi + élőváltozó-elemzés
  perf.ts         port/késleltetési költségmodell + online tanított futásidejű előrejelző
  cost.ts         cost(R;T) = we·eq(R;T) + wp·perf(R) ULP/bit metrikákkal
  testcases.ts    dinamikus teszt-adatbázis (véletlen, adversariális, ellenpéldák)
  mcmc.ts         Metropolis–Hastings szuperoptimalizáló (5 javaslati mag)
  mcts.ts         PUCT/MCTS szintézis online policy- és value-modellekkel
  enumerative.ts  kétirányú meet-in-the-middle enumeráció
  smt.ts          szimbolikus végrehajtás → CNF bit-blastolás → DPLL ekvivalencia
  egraph.ts       e-gráf, egyenlőségi telítés, pontos ILP-kinyerés
  hybrid.ts       párhuzamos orkesztrátor megosztott tudásbázissal
  tasks.ts        benchmark-csomag (11 kernel, egész és lebegőpontos)

runtime (Zig)      allokátorok, lock-free struktúrák, SIMD/COW tenzorok,
                   gyorstár-blokkolt mátrixszorzás, CUDA-stream GPU-kollektívák,
                   Futhark C-ABI kötései, szintetizált-kernel diszpécser
futhark_kernels    dot/saxpy/matmul/softmax/conv1d/stencil/FFT/linear-relu
docs               architecture.md, build_run.md, apis.md
ci                 ci.sh, fmt.sh, export_txt.sh, selftest.mjs
```

A Next.js alkalmazás a vezérlősík szerepét tölti be: hibrid keresések indítása, a motorok működésének megfigyelése, a Pareto-front vizsgálata, átírások bizonyítása a mellékelt SMT-megoldóval, egyenlőségi telítés futtatása a teljes egészében kirajzolt ILP-modellel, a tartósított ellenpéldák korpuszának böngészése és minden forrásartefaktum megtekintése.

Gyors indulás:

```bash
cp .env.example .env
npm install && npx drizzle-kit push && npm run build && npm run start
node ci/selftest.mjs          # optimalizáló önteszt (keresés + bizonyítások + eqsat)
cd runtime && zig build test  # futtatókörnyezet egységtesztek
cd futhark_kernels && ./build.sh
```

Lásd: docs/build_run.md, docs/architecture.md és docs/apis.md.
