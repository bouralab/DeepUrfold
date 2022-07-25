Forked from [Micak Stubb's Bl.ock](https://bl.ocks.org/micahstubbs/95fb1b19519ae63fe0bfb679fdf3510e), which was forked originally from [Matteo Abrate's Circle packing with hierarchical edge bundling](https://bl.ocks.org/nitaku/972a1a1ca93bb3da54505f3b0f3bb335)

Changes by Curran:

 * Upgraded to D3 v5.
 * Removed use of `attrs`, so it's just vanilla D3 now!

--- 

An iteration on [the previous example](http://bl.ocks.org/nitaku/72af4fb979e6689cffb3f7a031d9375f), this time leveraging [hierarchical edge bundling](https://www.win.tue.nl/vis1/home/dholten/papers/bundles_infovis.pdf) to show imports between packages in the historic [flare](http://flare.prefuse.org/) Actionscript visualization library. Direction is ignored.

---

this iteration on [Circle packing with hierarchical edge bundling](https://bl.ocks.org/nitaku/972a1a1ca93bb3da54505f3b0f3bb335) from [@matteoabrate](https://twitter.com/matteoabrate) runs `lebab.sh` on the output of the previous decaffeinated [iteration](https://github.com/micahstubbs/edge-bundling-experiments/tree/master/01).  then, some syntax is manually optimized for readability.

---

this iteration makes the code (subjectively) nice to work with

use [decaffeinate](https://decaffeinate-project.org/repl/#?useCS2=true&useJSModules=true&loose=false) to produce modern Javascript

that modern Javascript is then Prettier formatted https://prettier.io/


---

special thanks to [@currankelleher](https://twitter.com/currankelleher) who [tweeted](https://twitter.com/currankelleher/status/1030406374704873472) about this nice block and motivated me to fork it.
