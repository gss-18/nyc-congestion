"""Azure deployment architecture for the forecast service, as code (mingrammer/diagrams + Graphviz).

Writes docs/img/azure-architecture-light.svg and -dark.svg. The icons are the official Azure architecture icons that
ship with `diagrams`; they are embedded into the SVG as data URIs so GitHub renders the file without local paths.

Requires Graphviz (`brew install graphviz`). Run:  uv run python docs/architecture/azure_architecture.py
"""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path

import diagrams
from diagrams import Cluster, Diagram, Edge
from diagrams.azure.aimachinelearning import MachineLearning
from diagrams.azure.compute import ContainerApps, ContainerRegistries
from diagrams.azure.devops import ApplicationInsights
from diagrams.azure.identity import ManagedIdentities
from diagrams.azure.managementgovernance import Alerts
from diagrams.azure.security import KeyVaults
from diagrams.azure.storage import BlobStorage
from diagrams.custom import Custom
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.client import Client
from diagrams.onprem.network import Internet
from diagrams.onprem.vcs import Github
from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parents[1] / "img"
RESOURCES = Path(diagrams.__file__).parent.parent / "resources"
TMP = Path(tempfile.mkdtemp(prefix="arch-icons-"))


def light_icon(rel: str) -> str:
    """A light copy of a black monochrome icon (GitHub, client, cloud) so it stays visible on a dark background."""
    img = Image.open(RESOURCES / rel).convert("RGBA")
    r, g, b, a = img.split()
    rgb = ImageOps.invert(Image.merge("RGB", (r, g, b)))
    out = TMP / rel.replace("/", "_")
    Image.merge("RGBA", (*rgb.split(), a)).save(out)
    return str(out)


def mono_node(theme: str, cls, rel: str, label: str):
    """The stock node in the light theme; the same icon, inverted, in the dark theme."""
    return cls(label) if theme == "light" else Custom(label, light_icon(rel))

THEMES = {
    "light": dict(bg="white", text="#1f2328", sub="#59636e", edge="#57606a", accent="#0078d4", gate="#107c10",
                  azure_bg="#f3f8fd", azure_border="#0078d4", env_bg="#e6f1fb", gh_bg="#f6f8fa", gh_border="#8c959f",
                  shared_bg="#ffffff", shared_border="#afb8c1"),
    "dark": dict(bg="#0d1117", text="#e6edf3", sub="#9198a1", edge="#8b949e", accent="#4aa3ff", gate="#3fb950",
                 azure_bg="#0b1a2b", azure_border="#4aa3ff", env_bg="#10253c", gh_bg="#161b22", gh_border="#3d444d",
                 shared_bg="#0d1117", shared_border="#3d444d"),
}


def cluster_attr(bg: str, border: str, t: dict, style: str = "rounded") -> dict:
    return {"bgcolor": bg, "pencolor": border, "fontcolor": t["sub"], "fontsize": "14", "fontname": "Helvetica",
            "style": style, "margin": "22", "labeljust": "l"}


def draw(theme: str) -> Path:
    t = THEMES[theme]
    name = OUT / f"azure-architecture-{theme}"
    graph = {"bgcolor": t["bg"], "fontcolor": t["text"], "fontname": "Helvetica", "fontsize": "22",
             "pad": "0.5", "nodesep": "0.9", "ranksep": "1.3", "splines": "spline", "newrank": "true", "labelloc": "t",
             "label": "nyc-congestion forecast service on Azure\n\n"}
    node = {"fontcolor": t["text"], "fontname": "Helvetica", "fontsize": "13"}
    edge = {"color": t["edge"], "fontcolor": t["sub"], "fontname": "Helvetica", "fontsize": "12", "penwidth": "1.4"}

    def step(label: str, color: str | None = None, style: str = "solid") -> Edge:
        return Edge(label=label, color=color or t["edge"], fontcolor=color or t["sub"], style=style)

    with Diagram("", filename=str(name), outformat="svg", show=False, direction="LR",
                 graph_attr=graph, node_attr=node, edge_attr=edge):
        client = mono_node(theme, Client, "onprem/client/client.png", "API client\nGET /forecast")
        data = mono_node(theme, Internet, "onprem/network/internet.png",
                         "NYC TLC + Open-Meteo\nmonthly trips · weather")

        with Cluster("GitHub", graph_attr=cluster_attr(t["gh_bg"], t["gh_border"], t)):
            repo = mono_node(theme, Github, "onprem/vcs/github.png", "Repository\nnyc-congestion")
            ci = GithubActions("GitHub Actions\nruff · pytest · build")

        with Cluster("Azure  ·  resource group rg-nyc-forecast",
                     graph_attr=cluster_attr(t["azure_bg"], t["azure_border"], t)):
            acr = ContainerRegistries("Azure Container\nRegistry")
            blob = BlobStorage("Azure Blob Storage\napproved models · hourly tables")
            aml = MachineLearning("Azure Machine Learning\nMLflow runs · model registry")
            with Cluster("Container Apps environment", graph_attr=cluster_attr(t["env_bg"], t["azure_border"], t)):
                job = ContainerApps("Container Apps Job\nmonthly retraining")
                api = ContainerApps("Azure Container Apps\nforecast API (FastAPI)")
            shared = cluster_attr(t["shared_bg"], t["shared_border"], t, "dashed,rounded")
            with Cluster("Shared services  ·  identity, telemetry and alerts for both apps", graph_attr=shared):
                kv = KeyVaults("Azure Key Vault")
                mi = ManagedIdentities("Managed Identities")
                appi = ApplicationInsights("Application Insights")
                alerts = Alerts("Azure Monitor alerts")

        # Release path
        repo >> step("1  push") >> ci >> step("2  image") >> acr >> step("3  new revision") >> api
        # Request path (minlen keeps the client in the left column, next to GitHub)
        client >> Edge(label="4  HTTPS", color=t["accent"], fontcolor=t["accent"], minlen="2") >> api
        api >> step("5  load model") >> blob
        # Monthly retraining
        data >> Edge(label="A  download", color=t["edge"], fontcolor=t["sub"], minlen="2") >> job
        job >> step("B  log + register", style="dashed") >> aml
        job >> step("C  promote if WAPE < 15% and PSI < 0.2", t["gate"]) >> blob
        # Shared services sit in the right-hand column (invisible edges only fix their position).
        blob >> Edge(style="invis") >> appi >> Edge(label="alert rules", style="dotted", color=t["edge"],
                                                    fontcolor=t["sub"]) >> alerts
        aml >> Edge(style="invis") >> mi >> Edge(label="secrets", style="dotted", color=t["edge"],
                                                 fontcolor=t["sub"]) >> kv

    return name.with_suffix(".svg")


def embed_images(svg_path: Path) -> None:
    """Replace local icon file paths with base64 data URIs so the SVG is self-contained."""
    svg = svg_path.read_text()

    def inline(m: re.Match) -> str:
        path = Path(m.group(2))
        mime = "image/png" if path.suffix == ".png" else "image/svg+xml"
        return f'{m.group(1)}="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"'

    svg = re.sub(r'(xlink:href|href)="(/[^"]+\.(?:png|svg))"', inline, svg)
    svg_path.write_text(svg)


def main() -> None:
    for theme in THEMES:
        path = draw(theme)
        embed_images(path)
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
