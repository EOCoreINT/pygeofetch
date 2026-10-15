#!/usr/bin/env bash
# Build every language into build/<lang>/ and a root index that redirects by browser language.
set -euo pipefail
cd "$(dirname "$0")/.."
LANGS="en fr es pt de zh_CN ar"
for l in $LANGS; do
  echo "== $l"; DOCS_LANG=$l python3 -m sphinx -b html -W --keep-going -q -D language=$l . build/$l
done
cat > build/index.html <<'HTML'
<!doctype html><meta charset="utf-8"><title>pygeofetch</title>
<script>
var ok=["en","fr","es","pt","de","zh_CN","ar"],l="en";
try{l=localStorage.getItem("gf-lang")||"";}catch(e){}
if(ok.indexOf(l)<0){var n=(navigator.language||"en").replace("-","_");
 l=ok.indexOf(n)>=0?n:(ok.indexOf(n.split("_")[0])>=0?n.split("_")[0]:(n.indexOf("zh")==0?"zh_CN":"en"));}
location.replace(l+"/index.html");
</script><a href="en/index.html">pygeofetch documentation</a>
HTML
