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
