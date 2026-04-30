(function () {
  function renderMermaid() {
    if (typeof mermaid === "undefined") {
      return;
    }

    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default",
    });

    const nodes = document.querySelectorAll("pre.mermaid, div.mermaid");
    if (!nodes.length) {
      return;
    }

    nodes.forEach(function (node) {
      const code = node.querySelector("code");
      if (code) {
        node.textContent = code.textContent;
      }
      node.removeAttribute("data-processed");
    });

    mermaid.run({ nodes: nodes });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(renderMermaid);
  } else {
    document.addEventListener("DOMContentLoaded", renderMermaid);
  }
})();
