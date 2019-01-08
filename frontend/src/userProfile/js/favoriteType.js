import React from "react";
import {Chart, Geom, Axis, Tooltip, Coord} from "bizcharts";
import DataSet from "@antv/data-set";

class FavoriteType extends React.Component {
    render() {
        const {DataView} = DataSet;
        const data = [{type: "爱情", num: 90}, {type: "喜剧", num: 40}, {type: "动作", num: 120}, {type: "科幻", num: 426}, {
            type: "纪实",
            num: 30
        }, {type: "艺术", num: 0}, {type: "恐怖", num: 120}, {type: "剧情", num: 150}, {type: "冒险", num: 100}, {
            type: "动画",
            num: 50
        }, {type: "战争", num: 108}];
        const dv = new DataView().source(data);
        dv.transform({
            type: "fold",
            fields: ["num"],
            // 展开字段集
            key: "user",
            // key字段
            value: "score" // value字段
        });
        const cols = {
            score: {
                min: 0,
                max: 500
            }
        };
        return (
            <div>
                <Chart
                    height={window.innerHeight}
                    data={dv}
                    padding={[20, 20, 95, 20]}
                    scale={cols}
                    forceFit
                >
                    <Coord type="polar" radius={0.9}/>
                    <Axis
                        name="type"
                        line={null}
                        tickLine={null}
                        grid={{
                            lineStyle: {
                                lineDash: null
                            },
                            hideFirstLine: false
                        }}
                    />
                    <Tooltip/>
                    <Axis
                        name="score"
                        line={null}
                        tickLine={null}
                        grid={{
                            type: "polygon",
                            lineStyle: {
                                lineDash: null
                            },
                            alternateColor: "rgba(0, 0, 0, 0.04)"
                        }}
                    />
                    {/*<Legend name="user" marker="circle" offset={30} visible={false}/>*/}
                    <Geom type="area" position="type*score" color="user"/>
                    <Geom type="line" position="type*score" color="user" size={2}/>
                    <Geom
                        type="point"
                        position="type*score"
                        color="user"
                        shape="circle"
                        size={4}
                        style={{
                            stroke: "#fff",
                            lineWidth: 1,
                            fillOpacity: 1
                        }}
                    />
                </Chart>
            </div>
        );
    }
}
export default FavoriteType
