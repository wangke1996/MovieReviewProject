import React, {Component} from "react";
import {Chart, Geom, Axis, Tooltip, Legend, Label} from "bizcharts";
import moment from 'moment'
import DataSet from "@antv/data-set";
import Brush from "@antv/g2-brush";

function getComponent(data, pubDate) {
    const ds = new DataSet();
    const dv = ds.createView("reviewTrend").source(data).transform({
        type: "filter", callback(row) {
            let pubYear, pubMonth, pubDay;
            [pubYear, pubMonth, pubDay] = pubDate.split('-');
            let startDay = [(parseInt(pubYear) - 1).toString(), pubMonth, pubDay].join('-');
            return row.日期 > startDay;/*row.num > 2;*/
        }
    }).transform({type: "fold", key: "scoreType", value: "评分", fields: ["当日评分", "累计评分"]});
    const scale = {
        日期: {
            type: "time",
        },
        评分: {
            min: 0,
            max: 10,
            tickInterval: 1
        }
    };
    let chart;

    class RenderChart extends React.Component {
        componentDidMount() {
            new Brush({
                canvas: chart.get("canvas"),
                style: {
                    fill: "#ccc",
                    fillOpacity: 0.4
                },
                chart
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
                        height={400}
                        data={dv}
                        // padding={[60, 30, 30]}
                        scale={scale}
                        onGetG2Instance={g2Chart => {
                            g2Chart.animate(false);
                            chart = g2Chart;
                        }}
                        forceFit
                    >
                        <Tooltip/>
                        <Axis/>
                        <Legend position="top"/>
                        <Geom
                            type="line"
                            position="日期*评分"
                            color="scoreType"
                            shape={['scoreType', d => {
                                return d === '当日评分' ? 'hv' : 'spline';
                            }]}
                            size={2}
                        />
                    </Chart>
                </div>
            );
        }
    }

    return RenderChart;
}

class ScoreTrend extends Component {
    dataPreprocess() {
        let data = [];
        let reviewNum = 0;
        let totalScore = 0;
        this.props.reviewsTrendData.forEach((d, i) => {
            reviewNum += d.num;
            totalScore += d.num * d.rate * 2;
            data[i] = {
                '日期': d.time,
                '当日评分': Math.round(2 * d.rate * 100) / 100,
                '累计评分': Math.round(totalScore / reviewNum * 100) / 100,
                '评论数': d.num
            };
        });
        return data;
    }

    random_data() {
        let year = 2019, month = 2, startDay = 5, endDay = 28;
        let endTime = moment().format('YYYY-MM-DD');
        let nowMonth = parseInt(endTime.slice(5, 7));
        let nowDay = parseInt(endTime.slice(8, 10));
        if (nowMonth == month && nowDay < endDay)
            endDay = nowDay;
        let data = [
            {
                name: '累计评分',
            },
            {
                name: '当日评分',
            }
        ];
        let i = 0;
        let day;
        let averageScore = 0;
        for (day = startDay; day <= endDay; day++, i++) {
            let time = year + '-' + ('00' + month).slice(-2) + ('00' + day).slice(-2);
            let score = Math.round(100 * (Math.random() * 3 + 6)) / 100;
            averageScore += score;
            // let average_score = Math.round(100*(Math.random() * 5))/100;
            data[i] = {'日期': time, '当日评分': score, '累计评分': Math.round(100 * averageScore / (i + 1)) / 100};
        }
        return data;
    }

    render() {
        const data = this.dataPreprocess();
        const RenderChart = getComponent(data, this.props.pubDate);
        // const data = this.random_data();
        return (
            <div id='scoreTrend'>
                <header>
                    <h2>评分变化趋势</h2>
                    <span className="byline">总体评分<span
                        className="emphatic">{data[data.length - 1]['累计评分']}</span>/10</span>
                </header>
                <RenderChart/>
            </div>
        );
    }
}

ScoreTrend.defaultProps = {
    pubDate: '1900-01-01',
    reviewsTrendData: []
};
export default ScoreTrend;
