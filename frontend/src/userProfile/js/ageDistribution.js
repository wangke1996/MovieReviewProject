import React, {Component} from "react";
import {Chart, Geom, Axis, Tooltip, Legend, Shape, Util} from "bizcharts";

class AgeDistribution extends Component {
    render() {
        function getRectPath(points) {
            const path = [];

            for (let i = 0; i < points.length; i++) {
                const point = points[i];

                if (point) {
                    const action = i === 0 ? "M" : "L";
                    path.push([action, point.x, point.y]);
                }
            }

            const first = points[0];
            path.push(["L", first.x, first.y]);
            path.push(["z"]);
            return path;
        }

        function getFillAttrs(cfg) {
            const defaultAttrs = Shape.interval;
            const attrs = Util.mix(
                {},
                defaultAttrs,
                {
                    fill: cfg.color,
                    stroke: cfg.color,
                    fillOpacity: cfg.opacity
                },
                cfg.style
            );
            return attrs;
        }

        Shape.registerShape("interval", "waterfall", {
            draw(cfg, container) {
                const attrs = getFillAttrs(cfg);
                let rectPath = getRectPath(cfg.points);
                rectPath = this.parsePath(rectPath);
                const interval = container.addShape("path", {
                    attrs: Util.mix(attrs, {
                        path: rectPath
                    })
                });

                if (cfg.nextPoints) {
                    let linkPath = [
                        ["M", cfg.points[2].x, cfg.points[2].y],
                        ["L", cfg.nextPoints[0].x, cfg.nextPoints[0].y]
                    ];

                    if (cfg.nextPoints[0].y === 0) {
                        linkPath[1] = ["L", cfg.nextPoints[1].x, cfg.nextPoints[1].y];
                    }

                    linkPath = this.parsePath(linkPath);
                    container.addShape("path", {
                        attrs: {
                            path: linkPath,
                            stroke: "rgba(0, 0, 0, 0.45)",
                            lineDash: [4, 2]
                        }
                    });
                }

                return interval;
            }
        });
        const data = [
            {
                age: "?~1960",
                filmNums: 50
            },
            {
                age: "1960~1970",
                filmNums: 100
            },
            {
                age: "1970~1980",
                filmNums: 200
            },
            {
                age: "1980~1990",
                filmNums: 284
            },
            {
                age: "1990~2000",
                filmNums: 200
            },
            {
                age: "2000~2010",
                filmNums: 300
            },
            {
                age: "2010~2019",
                filmNums: 100
            },
            {
                age: "总计",
                filmNums: 1234
            }
        ];

        for (let i = 0; i < data.length; i++) {
            const item = data[i];

            if (i > 0 && i < data.length - 1) {
                if (Util.isArray(data[i - 1].filmNums)) {
                    item.filmNums = [
                        data[i - 1].filmNums[1],
                        item.filmNums + data[i - 1].filmNums[1]
                    ];
                } else {
                    item.filmNums = [data[i - 1].filmNums, item.filmNums + data[i - 1].filmNums];
                }
            }
        }
        return (

            <div className={"6u " + this.props.flag} id="ageDistribution">
                <header>
                    <h2>是个<span className="emphatic">怀旧</span>的影迷</h2>
                    <span className="byline">看了<span className="emphatic">834</span>部上个世纪的电影</span>
                </header>
                <Chart height={400} data={data} forceFit>
                    <Legend
                        custom={true}
                        clickable={false}
                        items={[
                            {
                                value: "观影的年代分布",
                                marker: {
                                    symbol: "square",
                                    fill: "#1890FF",
                                    radius: 5
                                }
                            },
                            {
                                value: "累计",
                                marker: {
                                    symbol: "square",
                                    fill: "#8c8c8c",
                                    radius: 5
                                }
                            }
                        ]}
                    />
                    <Axis name="age"/>
                    <Axis name="filmNums"/>
                    <Tooltip/>
                    <Geom
                        type="interval"
                        position="age*filmNums"
                        color={[
                            "age",
                            age => {
                                if (age === "总计") {
                                    return "rgba(0, 0, 0, 0.65)";
                                }

                                return "#1890FF";
                            }
                        ]}
                        tooltip={[
                            "age*filmNums",
                            (age, filmNums) => {
                                if (Util.isArray(filmNums)) {
                                    return {
                                        name: "观影数量",
                                        value: filmNums[1] - filmNums[0]
                                    };
                                }

                                return {
                                    name: "总观影数量",
                                    value: filmNums
                                };
                            }
                        ]}
                        shape="waterfall"
                    />
                </Chart>
            </div>
        );
    }
}

AgeDistribution.defaultProps = {flag: ''};
export default AgeDistribution
