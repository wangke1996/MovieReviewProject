import React, {Component} from "react";
import {Chart, Geom, Axis, Tooltip, Coord} from "bizcharts";
import DataSet from "@antv/data-set";
import {Rate} from 'antd';
import CountUp from 'react-countup';

class RateDistribution extends Component {
    constructor(props) {
        super(props);
        this.props.flag = "";
    }

    render() {
        const data = [
            {
                rate: "1 star",
                reviewNums: 264
            },
            {
                rate: "2 star",
                reviewNums: 180
            },
            {
                rate: "3 star",
                reviewNums: 460
            },
            {
                rate: "4 star",
                reviewNums: 210
            },
            {
                rate: "5 star",
                reviewNums: 120
            }
        ];
        const ds = new DataSet();
        const dv = ds.createView().source(data);
        dv.source(data).transform({
            type: "sort",

            //callback(a, b) {
            // 排序依据，和原生js的排序callback一致
            //return a.population - b.population > 0;
            //}
        });
        return (
            <div className={"6u "+this.props.flag}  id="rateDistribution">
                <header>
                    <h2>平均给分 <Rate disabled allowHalf defaultValue={5 * 6 / 10}/><span
                        className="emphatic"><CountUp
                        className="custom-count"
                        start={10.0}
                        end={6.0}
                        decimals={1}
                        duration={4}
                        useEasing={true}
                        redraw={true}
                    /></span></h2>
                    <span className="byline">真的很<span className="emphatic">严格</span></span>
                </header>
                <Chart height={400} data={dv} forceFit>
                    <Coord transpose/>
                    <Axis
                        name="rate"
                        label={{
                            offset: 12
                        }}
                    />
                    <Axis name="reviewNums"/>
                    <Tooltip/>
                    <Geom type="interval" position="rate*reviewNums"/>
                </Chart>
            </div>

        );
    }
}

export default RateDistribution