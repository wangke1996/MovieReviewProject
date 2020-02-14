import React, {Component} from 'react';
import ReactSpeedometer from "react-d3-speedometer";
import CountUp from 'react-countup';

class TotalNum extends Component {
    render() {
        return (
            <div className={"6u " + this.props.flag} id="totalNum">
                <header name={this.props.flag.toString()}>
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
        )
    }
}

TotalNum.defaultProps = {
    flag: "",
};
export default TotalNum