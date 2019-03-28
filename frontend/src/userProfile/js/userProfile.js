import React, {Component} from 'react';
import {Tag, Affix, Avatar} from 'antd'
import RateDistribution from './rateDistribution'
import FavoriteType from './favoriteType'
import ReviewList from './reviewList'
import AgeDistribution from './ageDistribution'
import ActorCard from './actorCard'
import TotalNum from './totalNum'
import '../css/userProfile.css';

class UserProfile extends Component {
    constructor(props) {
        super(props);
        props.colors = ["magenta", "red", "volcano", "orange", "gold", "lime"];
        props.tagJumpIds = ["totalNum", "rateDistribution", "favoriteType", "reviewList", "ageDistribution", "favoriteActor"];
        this.state = {
            // flagTotalNum: true,
            // flagRateDistribution: true,
            // flagFavoriteType: true,
            // flagReviewList: true,
            // flagAgeDistribution: true,
            // flagFavoriteActor: true
            flags: Array(6).fill(""),
            height: 0
        };
        this.updateWindowDimensions = this.updateWindowDimensions.bind(this);
    }

    componentDidMount() {
        this.updateWindowDimensions();
        window.addEventListener('resize', this.updateWindowDimensions);
    }

    componentWillUnmount() {
        window.removeEventListener('resize', this.updateWindowDimensions);
    }

    updateWindowDimensions() {
        this.setState({height: window.innerHeight, flags: this.state.flags});
    }

    handleTagClick(i) {
        const flags = new Array(6).fill("");
        flags[i] = "focused";
        this.setState({flags: flags, height: this.state.height});
    }

    renderTags(i, text) {
        return (
            <Tag color={this.props.colors[i]}><a href={"#" + this.props.tagJumpIds[i]}
                                                 onClick={() => this.handleTagClick(i)}
                                                 className="scrolly">{text}</a></Tag>
        )
    }

    render() {
        let prefix = process.env.PUBLIC_URL;
        return (

            <div id="Content" className="UserProfile">
                <div id="banner">
                    <h2>Hi! 欢迎使用 <strong>用户画像</strong>功能.</h2>
                    <span className="byline">
                        上传用户的影评，挖掘用户性格特点和电影审美
                    </span>
                    <hr/>
                </div>
                <div className="wrapper style1 align-center">
                    <section id="UserInfo">
                        <header>
                            <Affix offsetTop={this.state.height / 2}>
                                <Avatar src={prefix+"/source/images/avatar/SunWuKong.jpg"}
                                        size="large"/>
                                <div>
                                    {this.renderTags(0, "电影达人")}
                                    {this.renderTags(1, "真的很严格")}
                                    {this.renderTags(2, "科幻迷")}
                                    {this.renderTags(3, "剧情控")}
                                    {this.renderTags(4, "怀旧")}
                                    {this.renderTags(5, "章金莱")}
                                    {/*<Tag color="magenta"><a href={"#totalNum"} className="scrolly">电影达人</a></Tag>*/}
                                    {/*<Tag color="red"><a href={"#reviewList"} className="scrolly">剧情控</a></Tag>*/}
                                    {/*<Tag color="volcano"><a href={"#rateDistribution"}*/}
                                    {/*className="scrolly">真的很严格</a></Tag>*/}
                                    {/*<Tag color="orange"><a href={"#ageDistribution"} className="scrolly">怀旧</a></Tag>*/}
                                    {/*<Tag color="gold"><a href={"#favoriteType"} className="scrolly">科幻迷</a></Tag>*/}
                                    {/*<Tag color="lime"><a href={"#favoriteActor"} className="scrolly">玛丽莲梦露</a></Tag>*/}
                                </div>
                            </Affix>
                        </header>
                    </section>
                    <section id="AbstractGraphs">
                        <div className="AbstractGraph">
                            <div className="row">
                                <TotalNum flag={this.state.flags[0]}/>
                                <RateDistribution flag={this.state.flags[1]}/>
                            </div>
                            <div className="row">
                                <FavoriteType flag={this.state.flags[2]}/>
                                <ReviewList flag={this.state.flags[3]}/>
                            </div>
                            <div className="row">
                                <AgeDistribution flag={this.state.flags[4]}/>
                                <ActorCard flag={this.state.flags[5]}/>
                            </div>
                        </div>
                    </section>
                </div>
            </div>
        )
    }
}

export default UserProfile;
