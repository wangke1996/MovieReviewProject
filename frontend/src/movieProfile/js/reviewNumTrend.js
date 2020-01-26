import React, {Component} from "react";
import {Chart, Geom, Axis, Tooltip, Guide, Label} from "bizcharts";
import DataSet from "@antv/data-set";

import Brush from "@antv/g2-brush";
import {getMovieReviewsTrend} from "../../libs/getJsonData";
import LoadingSpin from "../../common/js/loadingSpin";
import ScoreTrend from './scoreTrend'

function getComponent(data, pubDate) {
    const ds = new DataSet();
    const dv = ds
        .createView("reviewTrend")
        .source(data)
        .transform({
            type: "filter",
            callback(row) {
                let pubYear, pubMonth, pubDay;
                [pubYear, pubMonth, pubDay] = pubDate.split('-');
                let startDay = [(parseInt(pubYear) - 1).toString(), pubMonth, pubDay].join('-');
                return row.time > startDay;//row.num > 2;
            }
        });
    const scale = {
        num: {
            alias: "相关评论数量"
        },
        time: {
            alias: "日期"
        }
    };
    let chart;

    function markerTextPosition() {
        let maxNum = 0;
        let releaseDayNum = -1;
        data.forEach(d => {
            if (d['time'] === pubDate)
                releaseDayNum = d['num'];
            if (d['num'] > maxNum)
                maxNum = d['num'];
        });
        return (maxNum + releaseDayNum) / (2 * maxNum);
    }

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
            const {DataMarker} = Guide;
            return (
                <div>
                    <Chart
                        height={400}
                        data={dv}
                        scale={scale}
                        onGetG2Instance={g2Chart => {
                            chart = g2Chart;
                        }}
                        forceFit
                    >
                        <Tooltip/>
                        <Axis/>
                        <Geom type="interval" position="time*num" color="#e50000">
                            <Label content="num" offset={0} textStyle={{
                                textAlign: 'center', // 文本对齐方向，可取值为： start middle end
                                fill: '#404040', // 文本的颜色
                                fontSize: '1em', // 文本大小
                                fontWeight: 'bold', // 文本粗细
                                textBaseline: 'bottom' // 文本基准线，可取 top middle bottom，默认为middle
                            }}/>
                        </Geom>
                        <Guide>
                            <DataMarker
                                top={true}
                                direction={"downward"}
                                position={[pubDate, 0]}
                                content={'电影上映'}
                                style={{
                                    text: {
                                        textAlign: 'right',
                                        fontSize: 15,
                                        fontWeight: 'bold',
                                        offset: 0,
                                        textBaseline: 'middle' // 文本基准线，可取 top middle bottom，默认为middle
                                    },
                                    point: {},
                                    line: {
                                        lineWidth: 2
                                    }
                                }}
                                lineLength={400 * 0.6 * markerTextPosition()}

                            />
                        </Guide>
                    </Chart>
                </div>
            );
        }
    }

    return SliderChart;
}

class ReviewNumTrend extends Component {

    loadData(movieID) {
        getMovieReviewsTrend(movieID, (data) => {
            this.setState({
                reviewsTrendData: data,
                loadedFlag: true
            })
        });
    }

    constructor(props) {
        super(props);
        this.state = {
            reviewsTrendData: [],
            loadedFlag: false,
        };
        this.loadData(this.props.movieID);
    }

    render() {
        if (!this.state.loadedFlag)
            return (<LoadingSpin tip='数据在线爬取中，可能需要数分钟时间'/>);
        const SliderChart = getComponent(this.state.reviewsTrendData, this.props.pubDate);
        const data = this.state.reviewsTrendData;

        function totalReviewNum() {
            let total = 0;
            data.forEach(d => {
                total += d['num'];
            });
            return total;
        }

        return (
            <div>
                <div className='6u'>
                    <header>
                        <h2>每日评论数量</h2>
                        <span className="byline">累计评论<span className="emphatic">{totalReviewNum()}</span>条</span>
                    </header>
                    <SliderChart/>
                </div>
                <div className='6u'>
                    <ScoreTrend reviewsTrendData={data} pubDate={this.props.pubDate}/>
                </div>
            </div>
        );
    }
}

// class ReviewNumTrend extends Component {
//
//     render() {
//         function random_data() {
//             let year = 2019, month = 2, startDay = 5, endDay = 28;
//             const maxReviewNum = 10000;
//             let endTime = moment().format('YYYY-MM-DD');
//             let nowMonth = parseInt(endTime.slice(5, 7));
//             let nowDay = parseInt(endTime.slice(8, 10));
//             if (nowMonth == month && nowDay < endDay)
//                 endDay = nowDay;
//             let data = [
//                 {
//                     name: '累计评论数',
//                 },
//                 {
//                     name: '当日评论数',
//                 }
//             ];
//             let i = 0;
//             let day;
//             let totalReview = 0;
//             let timeSet = [];
//             for (day = startDay; day <= endDay; day++, i++) {
//                 let time = year + '-' + ('00' + month).slice(-2) + ('00' + day).slice(-2);
//                 let reviewNum = Math.round(Math.random() * maxReviewNum);
//                 totalReview += reviewNum;
//                 // let average_score = Math.round(100*(Math.random() * 5))/100;
//                 data[0][time] = totalReview;
//                 data[1][time] = reviewNum;
//                 timeSet[i] = time;
//             }
//             return {'data': data, 'field': timeSet, 'total': totalReview};
//         }
//
//         const rand_data = random_data();
//         const ds = new DataSet();
//         const dv = ds.createView().source(rand_data.data);
//         dv.transform({
//             type: "fold",
//             fields: rand_data.field,
//             // 展开字段集
//             key: "time",
//             // key字段
//             value: "reviews" // value字段
//         });
//         return (
//             <div className="6u" id="reviewNumTrend">
//                 <header>
//                     <h2>每日评论数量</h2>
//                     <span className="byline">累计评论<span className="emphatic">{rand_data.total}</span>条</span>
//                 </header>
//                 <Chart height={400} data={dv} forceFit>
//                     <Legend/>
//                     <Axis name="time"/>
//                     <Axis name="reviews"/>
//                     <Tooltip/>
//                     <Geom
//                         type="interval"
//                         position="time*reviews"
//                         color={"name"}
//                         style={{
//                             stroke: "#fff",
//                             lineWidth: 1
//                         }}
//                     >
//                         <Label content="reviews" offset={0} textStyle={{
//                             textAlign: 'center', // 文本对齐方向，可取值为： start middle end
//                             fill: '#404040', // 文本的颜色
//                             fontSize: '1em', // 文本大小
//                             fontWeight: 'bold', // 文本粗细
//                             textBaseline: 'bottom' // 文本基准线，可取 top middle bottom，默认为middle
//                         }}/>
//                     </Geom>
//                 </Chart>
//             </div>
//     );
//     }
//     }


export default ReviewNumTrend;
