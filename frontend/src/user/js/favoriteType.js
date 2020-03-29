import React from "react";
import {Chart, Geom, Axis, Tooltip, Coord} from "bizcharts";
import DataSet from "@antv/data-set";

class FavoriteType extends React.Component {
    state = {
        favoriteType: '',
        favoriteNum: 0,
    };
    updateFavorite = () => {
        const {data} = this.props;
        let favoriteNum = 0, favoriteType = '';
        data.forEach(d => {
            if (d.num > favoriteNum) {
                favoriteNum = d.num;
                favoriteType = d.type;
            }
        });
        this.setState({favoriteNum, favoriteType});
    };

    componentDidMount() {
        this.updateFavorite()
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (prevProps.data !== this.props.data)
            this.updateFavorite();
    }

    generateHint = () => {
        const {favoriteType, favoriteNum} = this.state;
        return favoriteNum > 100 ?
            <span className="byline">资深<span className="emphatic">{favoriteType}迷</span></span> :
            favoriteNum > 50 ? <span className="byline"><span className="emphatic">{favoriteType}片</span>爱好者</span> :
                <span className="byline">刚入门<span className="emphatic">{favoriteType}片</span></span>;

    };

    render() {
        const {data, flag, text} = this.props;
        const {favoriteNum, favoriteType} = this.state;
        const {DataView} = DataSet;
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
                max: favoriteNum
            }
        };
        return (
            <div className={"6u " + flag} id="favoriteType">
                <header>
                    <h2>看了<span className="emphatic">{favoriteNum}</span>部{favoriteType}电影</h2>
                    {/*{this.generateHint()}*/}
                    <span className='byline'>{text}</span>
                </header>
                <div className='center'>
                    <Chart
                        height={400}
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
            </div>
        );
    }
}

export default FavoriteType
