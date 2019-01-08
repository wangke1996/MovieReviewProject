import React, {Component} from 'react';
import ReactSpeedometer from "react-d3-speedometer";
import CountUp from 'react-countup';
import {Rate} from 'antd';
import RateDistribution from './rateDistribution'
import FavoriteType from './favoriteType'
import ReviewList from './reviewList'
import AgeDistribution from './ageDistribution'
import ActorCard from './actorCard'
class AbstractGraphs extends Component {
    render() {
        return (
            <div className="AbstractGraph align-center">
                <div className="row">
                    <div className="6u" id="TotalNum">
                        <header>
                            <h2>看过<span className="emphatic">
                                <CountUp
                                    className="custom-count"
                                    start={0}
                                    end={1234}
                                    duration={4}
                                    useEasing={true}
                                    redraw={true}
                                /></span>
                                部电影</h2>
                            <span className="byline">
                                电影<span className="emphatic">达人</span>
                        </span>
                        </header>
                        <div style={{width: "500px", height: "300px"}}>
                            <ReactSpeedometer
                                fluidWidth
                                minValue={0}
                                maxValue={1500}
                                segments={10}
                                startColor="blue"
                                endColor="green"
                                needleTransitionDuration={4000}
                                needleTransition="easeElastic"
                                value={1234}
                            />
                        </div>
                    </div>
                    <div className="6u" id="rateDistribution">
                        <header>
                            <h2>平均给分 <Rate disabled allowHalf defaultValue={5 * 6.8 / 10}/><span
                                className="emphatic">{6.8}</span></h2>
                            <span className="byline">真的很<span className="emphatic">严格</span></span>
                        </header>
                        <RateDistribution/>
                    </div>
                </div>
                <div className="row">
                    <div className="6u" id="favoriteType">
                        <header>
                            <h2>看了<span className="emphatic">426</span>部科幻电影</h2>
                            <span className="byline">资深<span className="emphatic">科幻迷</span></span>
                        </header>
                        <FavoriteType/>
                    </div>
                    <div className="6u" id="mostCare">
                        <header>
                            <h2>对<span className="emphatic">剧情</span>最为挑剔</h2>
                            <span className="byline">对电影{"剧情"}的评价中，负面评价多达<span className="emphatic">80%</span></span>
                        </header>
                        <ReviewList/>
                    </div>
                </div>
                <div className="row">
                    <div className="6u" id="timeDistribution">
                        <header>
                            <h2>是个<span className="emphatic">怀旧</span>的影迷</h2>
                            <span className="byline">看了<span className="emphatic">834</span>部上个世纪的电影</span>
                        </header>
                        <AgeDistribution/>
                    </div>
                    <div className="6u" id="favoriteActor">
                        <header>
                            <h2>最喜欢的演员是<span className="emphatic">玛丽莲梦露</span></h2>
                            <span className="byline">看了<span className="emphatic">32</span>部她主演的电影，可以说是铁杆粉丝了</span>
                        </header>
                        <ActorCard/>
                    </div>
                </div>
            </div>
        )
    }
}

export default AbstractGraphs