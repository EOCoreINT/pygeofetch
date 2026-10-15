/* Sidebar accordion: each caption becomes a collapsible group; only the group
   that contains the current page starts open. */
document.addEventListener("DOMContentLoaded", function () {
  var tree = document.querySelector(".sidebar-tree");
  if (!tree) return;
  var caps = Array.prototype.slice.call(tree.querySelectorAll(":scope > p.caption"));
  caps.forEach(function (cap) {
    var list = cap.nextElementSibling;
    if (!list || list.tagName !== "UL") return;
    var group = document.createElement("div");
    group.className = "gf-group";
    cap.parentNode.insertBefore(group, cap);
    group.appendChild(cap);
    group.appendChild(list);
    var hasCurrent = !!list.querySelector("li.current, li.current-page, a.current");
    group.classList.toggle("gf-open", hasCurrent);
    cap.setAttribute("tabindex", "0");
    cap.setAttribute("aria-expanded", hasCurrent ? "true" : "false");
    function toggle() {
      var open = !group.classList.contains("gf-open");
      group.classList.toggle("gf-open", open);
      cap.setAttribute("aria-expanded", open ? "true" : "false");
    }
    cap.addEventListener("click", toggle);
    cap.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
  });
  var cur = tree.querySelector("li.current-page > a, a.current");
  if (cur && cur.scrollIntoView) { cur.scrollIntoView({ block: "center" }); }
});

/* Subtle motion: fade/slide-in for figures, cards and tables as they scroll into view,
   plus click-to-enlarge for figures. Respects prefers-reduced-motion. */
document.addEventListener("DOMContentLoaded", function () {
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var targets = document.querySelectorAll("article figure, article .sd-card, article table.docutils, article .admonition, article div[class*='highlight']");
  if (!reduce && "IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("gf-in"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    targets.forEach(function (t) { t.classList.add("gf-reveal"); io.observe(t); });
  }
  var box;
  function close() { if (box) { box.remove(); box = null; } }
  document.querySelectorAll("article figure img").forEach(function (img) {
    img.style.cursor = "zoom-in";
    img.addEventListener("click", function () {
      close();
      box = document.createElement("div"); box.className = "gf-lightbox";
      var big = document.createElement("img"); big.src = img.currentSrc || img.src; big.alt = img.alt;
      box.appendChild(big); box.addEventListener("click", close); document.body.appendChild(box);
    });
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") close(); });
});
document.addEventListener("click",function(e){var a=e.target.closest&&e.target.closest(".gf-lang-menu a");if(a){try{localStorage.setItem("gf-lang",a.dataset.lang)}catch(_){}}});
