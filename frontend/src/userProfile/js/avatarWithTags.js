import React, {Component} from 'react';
import {Avatar, Tag} from 'antd';

class AvatarWithTags extends Component {
    render() {
        return (
            <div>
                <Avatar src="https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png" size="large"/>
                <div>
                    <Tag color="magenta">电影达人</Tag>
                    <Tag color="red">剧情控</Tag>
                    <Tag color="volcano">斯皮尔伯格</Tag>
                    <Tag color="orange">真的很严格</Tag>
                    <Tag color="gold">怀旧</Tag>
                    <Tag color="lime">科幻迷</Tag>
                    <Tag color="green">玛丽莲梦露</Tag>
                </div>
            </div>
        )
    }
}

export default AvatarWithTags