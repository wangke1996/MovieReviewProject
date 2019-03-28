import React, {Component} from "react";
import {Chart, Geom, Axis, Tooltip, Legend, Label} from "bizcharts";
import moment from 'moment'

class ScoreTrend extends Component {

    render() {
        function random_data() {
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

        const data = random_data();
        const scale = {
            当日评分: {
                min: 0,
                max: 10
            },
            累计评分: {
                min: 0,
                max: 10
            }
        };
        let chartIns = null;
        return (
            <div className="6u" id="scoreTrend">
                <header>
                    <h2>评分变化趋势</h2>
                    <span className="byline">总体评分<span
                        className="emphatic">{data[data.length - 1]['累计评分']}</span>/10</span>
                </header>
                <Chart
                    height={400}
                    scale={scale}
                    forceFit
                    data={data}
                    onGetG2Instance={chart => {
                        chartIns = chart;
                    }}
                >

                    <Legend custom={true} attachLast={true}
                            allowAllCanceled={true}
                            items={[
                                {
                                    value: "当日评分",
                                    marker: {
                                        symbol: "hyphen",
                                        stroke: "#3182bd",
                                    }
                                },
                                {
                                    value: "累计评分",
                                    marker: {
                                        symbol: "circle",
                                        fill: "#fdae6b",
                                    }
                                }
                            ]}
                            onClick={ev => {
                                const item = ev.item;
                                const value = item.value;
                                const checked = ev.checked;
                                const geoms = chartIns.getAllGeoms();

                                for (let i = 0; i < geoms.length; i++) {
                                    const geom = geoms[i];

                                    if (geom.getYScale().field === value) {
                                        if (checked) {
                                            geom.show();
                                        } else {
                                            geom.hide();
                                        }
                                    }
                                }
                            }}
                    />
                    <Axis name="日期"/>
                    <Axis
                        name="累计评分"
                        grid={null}
                        label={{
                            textStyle: {
                                fill: "#fdae6b"
                            }
                        }}
                    />
                    <Tooltip/>
                    <Geom type='line' position="日期*当日评分" color="#3182bd" style={{
                        lineDash: [4, 4]
                    }} size={1} shape="hv"/>
                    <Geom
                        type="line"
                        position="日期*累计评分"
                        color="#fdae6b"
                        size={3}
                        shape={"smooth"}
                    >

                        <Label content="累计评分" offset={5}
                               textStyle={{
                                   textAlign: 'center', // 文本对齐方向，可取值为： start middle end
                                   fill: '#fdae6b', // 文本的颜色
                                   fontSize: '1em', // 文本大小
                                   fontWeight: 'bold', // 文本粗细
                                   textBaseline: 'bottom' // 文本基准线，可取 top middle bottom，默认为middle
                               }}/>
                    </Geom>
                    <Geom
                        type="point"
                        position="日期*累计评分"
                        color="#fdae6b"
                        size={5}
                        shape={"circle"}
                    />
                </Chart>
            </div>
        );
    }
}


export default ScoreTrend;
