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
import {SentimentProfile} from "./sentimentProfile";

class MovieProfile extends Component {
    loadData(movieID) {
        this.setState((state) => {
            state.data = {};
            state.loadedFlag = false;
            return state;
        });
        getMovieInfo(movieID, (data) => {
            this.setState({movieID, data, loadedFlag: true})
        });
    }

    state = {
        movieID: '-1',
        data: {},
        loadedFlag: false,
        useDefault: false
    };

    componentDidMount() {
        this.loadData(this.props.movieID);
    }

    render() {
        const {loadedFlag, data, movieID, useDefault} = this.state;
        if (!loadedFlag)
        // render nothing before data loaded
            return (<LoadingSpin/>);
        if (isEmpty(data)) {
            // data loaded, but is empty, request for default data
            this.setState((state) => {
                state.useDefault = true;
                return state;
            });
            this.loadData(this.props.defaultMovieID);
            return (<LoadingSpin/>)
        }
        if (useDefault) {
            message.info('无法找到指定电影的相关信息，已为您显示最近热门电影“' + data['title'] + '”', 10);
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
                                    <Summary data={data}/>
                                </div>
                                <div id='hotComments'>
                                    <HotComments movieID={movieID}/>
                                </div>
                            </div>
                            <div className='4u'>
                                <div id='basicInfo'>
                                    <BasicInfo data={data}/>
                                </div>
                                <div id='stills'>
                                    <Stills movieID={movieID}/>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="wrapper style1 align-center">
                    <div className="row">
                        <ReviewNumTrend movieID={movieID} pubDate={data['pubdate'] || data['pubdates']}/>
                        {/*<ScoreTrend/>*/}
                    </div>
                </div>
                {/*<div className="wrapper style1 align-center">*/}
                {/*    <div className="row">*/}
                <SentimentProfile id={movieID} type='movie'/>
                {/*</div>*/}
                {/*</div>*/}
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