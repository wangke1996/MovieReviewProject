import React, {Component} from "react";
import {Chart, Geom, Axis, Tooltip, Coord} from "bizcharts";
import DataSet from "@antv/data-set";

class RateDistribution extends Component {
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
            <div>
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