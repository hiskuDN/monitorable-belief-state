import { useEffect, useState } from "react";
import Demo from "./pages/Demo.jsx";
import Concealworld from "./pages/Concealworld.jsx";
import Storyworld from "./pages/Storyworld.jsx";
import Blogs from "./pages/Blogs.jsx";
import BlogPost from "./pages/BlogPost.jsx";

// Minimal hash router (no dep), robust for static hosting under any base path.
function useHash() {
  const [hash, setHash] = useState(() => window.location.hash || "#/");
  useEffect(() => {
    const on = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

function route(hash) {
  const path = hash.replace(/^#/, "") || "/";
  if (path.startsWith("/storyworld")) return { name: "storyworld" };
  // legacy alias kept so old links to the K-cell view still resolve
  if (path.startsWith("/concealworld")) return { name: "concealworld" };
  if (path === "/blogs" || path === "/blogs/") return { name: "blogs" };
  const m = path.match(/^\/blogs\/(.+?)\/?$/);
  if (m) return { name: "post", slug: m[1] };
  return { name: "demo" };
}

export default function App() {
  const hash = useHash();
  const r = route(hash);
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [hash]);

  return (
    <>
      <nav className="topnav">
        <div className="topnav-inner">
          <a className="brand" href="#/">
            Monitorable belief state <span>· demo</span>
          </a>
          <div className="topnav-links">
            <a href="#/" className={r.name === "demo" ? "active" : ""}>
              Demo 1 · Gridworld
            </a>
            <a href="#/concealworld" className={r.name === "concealworld" ? "active" : ""}>
              Demo 2 · Concealment
            </a>
            <a href="#/storyworld" className={r.name === "storyworld" ? "active" : ""}>
              Demo 3 · Storyworld
            </a>
          </div>
        </div>
      </nav>
      {r.name === "demo" && <Demo />}
      {r.name === "concealworld" && <Concealworld />}
      {r.name === "storyworld" && <Storyworld />}
      {r.name === "blogs" && <Blogs />}
      {r.name === "post" && <BlogPost slug={r.slug} />}
    </>
  );
}
