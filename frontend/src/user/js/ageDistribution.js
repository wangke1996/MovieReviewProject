import React, {Component} from "react";
import {Chart, Geom, Axis, Tooltip, Legend, Shape, Util} from "bizcharts";
import DataSet from "@antv/data-set";
import Brush from "@antv/g2-brush";


function getComponent(data, tickInterval = 1) {
    const ds = new DataSet();
    const dv = ds
        .createView("ageDistribution")
        .source(data);
    const scale = {
        count: {
            alias: "观影数量"
        },
        year: {
            tickInterval: tickInterval,
            alias: "上映年代"
        }
    };
    let chart;

    class SliderChart extends React.Component {
        componentDidMount() {
            new Brush({
                canvas: chart.get("canvas"),
                chart,
                type: "X",

                onBrushstart() {
                    chart.hideTooltip();
                },

                onBrushmove() {
                    chart.hideTooltip();
                }
            });
            chart.on("plotdblclick", () => {
                chart.get("options").filters = {};
                chart.repaint();
            });
        }

        render() {
            return (
                <div>
                    <Chart
                        data={dv}
                        scale={scale}
                        height={400}
                        onGetG2Instance={g2Chart => {
                            chart = g2Chart;
                        }}
                        forceFit
                    >
                        <Tooltip/>
                        <Axis/>
                        <Geom type="interval" position="year*count" color="#e50000"/>
                    </Chart>
                </div>
            );
        }
    }

    return SliderChart;
}

class AgeDistribution extends Component {
    render() {
        const {data, tag, text} = this.props;
        const SliderChart = getComponent(data,1);
        return (
            <div className={"6u " + this.props.flag} id="ageDistribution">
                <header>
                    <h2>是个<span className="emphatic">{tag}</span>的影迷</h2>
                    <span className="byline">{text}</span>
                </header>
                <SliderChart/>
            </div>
        );
    }
}

export default AgeDistribution
