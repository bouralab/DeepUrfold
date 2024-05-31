import os
import subprocess
import multiprocessing
import sys
from time import sleep
from pathlib import Path

import click
import pandas as pd
from joblib import Parallel, delayed
from requests_html import HTMLSession
from flask import Flask, render_template

flare=None
flare_links=None

app = Flask(__name__,
        static_url_path='/static',
        static_folder='./static',)
source_dir = Path(__file__).parent.resolve()

features = ['ss', 'charge', 'electrostatics', 'sfam', 'go_mf', 'go_bp', 'go_cc']

using_cath = False


@click.command()
@click.option('--flare_path', default=None, help='Path to flare csv file created from StochasticBlockModel.')
@click.option('--flare_links', default=None, help='Path to flare links csv file created from StochasticBlockModel.')
@click.option('--port', default=8000, help='Port to use')
@click.option('--save_svg', default=False, is_flag=True, help='save svg file')
@click.option('--feature', default=["sfam"], multiple=True, type=click.Choice(features+['all']), help='save svg file')
@click.option('--use_cath', default=False, is_flag=True, help='load CATH sbm')
def cli(flare_path=None, flare_links=None, port=8000, feature=["sfam"], save_svg=False, use_cath=False):
    run(flare_path=flare_path, flare_links=flare_links, port=port, feature=feature, save_svg=save_svg, use_cath=use_cath)

def run(flare_path=None, flare_links=None, port=8000, feature=["sfam"], save_svg=False, use_cath=False):
    global flare, using_cath
    flare = Path(flare_path).resolve() if flare_path is not None else Path("flare.csv").resolve()
    flare_links = Path(flare_links).resolve() if flare_links is not None else Path("flare_linksare.csv").resolve()

    using_cath = use_cath

    assert flare.exists(), "Flare file must exist, please create using StochasticBlockModel class"

    assert source_dir.exists(), f"Invalid app_name. Please check DeepUrfold.WebApp for available apps: {source_dir}"

    if not save_svg:
        app.run(
            port=port,
        )
    else:
        if not using_cath:
            path = Path("sbm_figures")
        else:
            path = Path("cath_sbm_figures")

        if 'all' in feature:
            feature = features

        needs_server = not all([(path/f"{f}.svg").exists() for f in feature])

        if needs_server:
            server = multiprocessing.Process(target=app.run, name="DeepUrfold Server", kwargs={"port":port})
            server.start()

            while not server.is_alive():
                sleep(1)
            #min(len(features), multiprocessing.cpu_count()-1)
        #Parallel(n_jobs=1)(delayed(download_svg)(f, port) for f in feature)

        try:
            for f in feature:
                download_svg(f, port)
        except Exception:
            raise
        finally:
            if needs_server:
                server.terminate()
                server.join()


def start_server(port):
    app.run(
        port=port,
    )

def download_svg(feature=["all"], port=8000):
    f = feature

    if not using_cath:
        path = Path("sbm_figures")
    else:
        path = Path("cath_sbm_figures")

    path.mkdir(exist_ok=True)

    svg_file = path / f"{f}.svg"
    svg_legend_file = path / f"{f}.legend.svg"

    if not svg_file.exists() or not svg_legend_file.exists():
        session = HTMLSession()
        r = session.get(f'http://127.0.0.1:{port}?feature={f}&save_svg=true')
        r.html.render(wait=4, sleep=4)

        svg = r.html.find('#deepurfold_div', first=True).find('svg', first=True)
        import pdb; pdb.set_trace()
        assert "circle" in r.html.find('#deepurfold_div', first=True).html
        assert "circle" in svg.html

        legend = svg.find('g.legendSequential', first=True)
        n_legend = len(list(legend.find("g.cell")))

        with svg_legend_file.open("w") as fh:
            print(f"<svg width='{svg.attrs['width']}' height='{svg.attrs['height']}'>{legend.html}</svg>", file=fh)

        with svg_file.open("w") as fh:
            print(svg.html.replace(legend.html, ""), file=fh)

    for img in [f"{f}.legend", f]:
        print("Running", img)
        if not (path/f"{img}.raw.png").exists():
            print(f"Converting svg to png: {img}.svg -> {img}.raw.png")
            p1 = subprocess.check_output(["inkscape", "-d", "3600", f"{img}.svg", "-o", f"{img}.raw.png"], cwd=path)
        else:
            print("Raw png exists")

        print("Why")  
        assert (path/f"{img}.raw.png").exists()

        print("Trimming?")

        if not (path/f'{img}.big.png').exists():
            print(f"Trim svg: {img}.raw.png -> {img}_big.png")
            subprocess.check_output(["magick", "convert", str(path/f'{img}.raw.png'), "-trim", str(path/f'{img}.big.png')])

        assert (path/f"{img}.big.png").exists()

        #Always redo this step
        print(f"Resize: {img}.big.png -> {img}.png")
        resize = "x1200" if "legend" not in img else "400x"
        subprocess.check_output(["magick", "convert", os.path.join(str(path), f'{img}.big.png'),
            "-resize", resize, "-quality", "600", "-unsharp", "0.25x0.25+8+0.065",
            "-dither", "None", "-interlace", "none", "-colorspace", "sRGB",
            #"-define png:compression-filter=5",
            #"-define png:compression-level=9",
            #"-define png:compression-strategy=1",
            os.path.join(str(path), f'{img}.png')])

    subprocess.check_output(["magick", "convert", "-transparent", "white", "+append", str(svg_legend_file.with_suffix(".png")), str(svg_file.with_suffix(".png")), str(svg_file.with_suffix(".withLegend.png"))])

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/mf')
def mf():
    mf = "sbm-mf.txt" if not using_cath else "cath-mf.txt"
    with open(mf) as f:
        mf_csv = f.read()
    return mf_csv

@app.route('/bp')
def bp():
    bp = "sbm-bp.txt" if not using_cath else "cath-bp.txt"
    with open(bp) as f:
        bp_csv = f.read()
    return bp_csv


@app.route('/cc')
def cc():
    cc = "sbm-cc.txt" if not using_cath else "cath-cc.txt"
    with open(cc) as f:
        cc_csv = f.read()
    return cc_csv

@app.route('/cath_mf')
def cath_mf():
    with open("cath-mf.txt") as f:
        mf_csv = f.read()
    return mf_csv

@app.route('/cath_bp')
def cath_bp():
    with open("cath-bp.txt") as f:
        bp_csv = f.read()
    return bp_csv


@app.route('/cath_cc')
def cath_cc():
    with open("cath-cc.txt") as f:
        cc_csv = f.read()
    return cc_csv

@app.route('/flare')
def flare():
    flare = "flare.csv" if not using_cath else "flare-cath.csv"
    with open(flare) as f:
        flare_csv = f.read()
    return flare_csv

@app.route('/flare_links')
def flare_links():
    with open("flare_links.csv") as f:
        flare_links = f.read()
    return flare_links

@app.route('/cath')
def cath():
    with open("flare-cath.csv") as f:
        cath = f.read()
    return cath



if __name__ == "__main__":
    cli()
