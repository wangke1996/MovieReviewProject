import React, {Component} from 'react';
import ReviewNumTrend from './reviewNumTrend'
// import ScoreTrend from './scoreTrend'
// import MovieInfo from './movieInfo'
import {getMovieInfo} from '../../libs/getJsonData'
import {isEmpty} from '../../libs/toolFunctions'
import {message} from "antd";
import Summary from './summary'
import HotComments from './hotComments'
import BasicInfo from './basicInfo'
import Stills from './stills'
import LoadingSpin from '../../common/js/loadingSpin'

class MovieProfile extends Component {
    loadData(movieID) {
        this.setState((state) => {
            state.data = {};
            state.loadedFlag = false;
            return state;
        });
        getMovieInfo(movieID, (data) => {
            this.setState((state) => {
                state.movieID = movieID;
                state.data = data;
                state.loadedFlag = true;
                return state;
            })
        });
    }

    constructor(props) {
        super(props);
        this.state = {
            movieID: '-1',
            data: {},
            loadedFlag: false,
            useDefault: false
        };
        this.loadData(this.props.movieID);
    }

    render() {
        if (!this.state.loadedFlag)
        // render nothing before data loaded
            return (<LoadingSpin/>);
        if (isEmpty(this.state.data)) {
            // data loaded, but is empty, request for default data
            this.setState((state) => {
                state.useDefault = true;
                return state;
            });
            this.loadData(this.props.defaultMovieID);
            return (<LoadingSpin/>)
        }
        if (this.state.useDefault) {
            message.info('无法找到指定电影的相关信息，已为您显示最近热门电影“' + this.state.data['title'] + '”', 10);
        }
        return (
            <div id="Content" className="MovieProfile">
                <div id="banner">
                    <h2>Hi! 欢迎使用 <strong>电影风评</strong>功能.</h2>
                    <span className="byline">
                        查询电影评价，跟踪最新电影评论趋势
                    </span>
                    <hr/>
                </div>
                <div className="wrapper style1">
                    <div className='container'>
                        <div className='row'>
                            <div className='8u'>
                                <div id='summary'>
                                    <Summary data={this.state.data}/>
                                </div>
                                <div id='hotComments'>
                                    <HotComments movieID={this.state.movieID}/>
                                </div>
                            </div>
                            <div className='4u'>
                                <div id='basicInfo'>
                                    <BasicInfo data={this.state.data}/>
                                </div>
                                <div id='stills'>
                                    <Stills movieID={this.state.movieID}/>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="wrapper style1 align-center">
                    <div className="row">
                        <ReviewNumTrend movieID={this.state.movieID} pubDate={this.state.data['pubdate']}/>
                        {/*<ScoreTrend/>*/}
                    </div>
                </div>
            </div>

        )
    }
}

MovieProfile.defaultProps = {
    defaultMovieID: '26266893',
    movieID: '26147417'//"1652592"
    // movieID: 'errorTest'
};
export default MovieProfile;