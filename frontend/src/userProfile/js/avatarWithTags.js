import React, {Component} from 'react';
import {Avatar, Tag, Affix} from 'antd';

class AvatarWithTags extends Component {
    constructor(props) {
        super(props);
        this.state = {height: 0};
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
        this.setState({height: window.innerHeight});
    }

    render() {
        return (
            <div className="align-center">
                <Affix offsetTop={this.state.height / 2}>
                    <Avatar src="https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png" size="large"/>
                    <div>
                        <Tag color="magenta"><a href={"#totalNum"} className="scrolly">电影达人</a></Tag>
                        <Tag color="red"><a href={"#reviewList"} className="scrolly">剧情控</a></Tag>
                        <Tag color="volcano"><a href={"#rateDistribution"} className="scrolly">真的很严格</a></Tag>
                        <Tag color="orange"><a href={"#ageDistribution"} className="scrolly">怀旧</a></Tag>
                        <Tag color="gold"><a href={"#favoriteType"} className="scrolly">科幻迷</a></Tag>
                        <Tag color="lime"><a href={"#favoriteActor"} className="scrolly">玛丽莲梦露</a></Tag>
                    </div>
                </Affix>
            </div>
        )
    }
}

export default AvatarWithTags