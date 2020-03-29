import React, {Component} from 'react';
import ReactSpeedometer from "react-d3-speedometer";
import CountUp from 'react-countup';

class TotalNum extends Component {
    render() {
        const {totalNum, text} = this.props;
        return (
            <div className={"6u " + this.props.flag} id="totalNum">
                <header>
                    <h2>看过<span className="emphatic">
                                <CountUp
                                    className="custom-count"
                                    start={0}
                                    end={totalNum}
                                    duration={10}
                                    useEasing={true}
                                    redraw={true}
                                /></span>
                        部电影</h2>
                    <span className="byline">{text}</span>
                </header>
                <div className='center' style={{width: "500px", height: "300px"}}>
                    <ReactSpeedometer
                        fluidWidth
                        minValue={0}
                        maxValue={Math.max(1000,totalNum)}
                        segments={10}
                        startColor="blue"
                        endColor="green"
                        needleTransitionDuration={10000}
                        needleTransition="easeElastic"
                        value={totalNum}
                    />
                </div>
            </div>
        )
    }
}

export default TotalNum