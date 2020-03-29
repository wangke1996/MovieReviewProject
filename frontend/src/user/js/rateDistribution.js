import React, {Component} from "react";
import {Chart, Geom, Axis, Tooltip, Coord} from "bizcharts";
import DataSet from "@antv/data-set";
import {Rate} from 'antd';
import CountUp from 'react-countup';

class RateDistribution extends Component {
    render() {
        const {averageScore, text, data} = this.props;
        const ds = new DataSet();
        const dv = ds.createView().source(data);
        dv.source(data).transform({
            type: "sort",
        });
        return (
            <div className={"6u " + this.props.flag} id="rateDistribution">
                <header>
                    <h2>平均给分 <Rate disabled allowHalf defaultValue={averageScore}/><span
                        className="emphatic"><CountUp
                        className="custom-count"
                        start={0.0}
                        end={averageScore}
                        decimals={1}
                        duration={10}
                        useEasing={true}
                        redraw={true}
                    /></span></h2>
                    <span className="byline">{text}</span>
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