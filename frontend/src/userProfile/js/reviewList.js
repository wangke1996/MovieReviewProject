import React, {Component} from 'react'
import {Comment, Tooltip, List, Rate} from 'antd';
import moment from 'moment';

class SingleComment extends Component {
    state = {
        full: false,
    };
    action = () => this.setState({full: !(this.state.full)});

    render() {
        const full = this.state.full;
        const actions = [<a onClick={this.action}>{full ? "简略信息" : "查看全文"}</a>];
        const content = full ? this.props.fullContent : this.props.content;
        const info = <span>作者：{this.props.author} | 影片：<strong>{this.props.filmName}</strong> | 打分：<Rate disabled
                                                                                                         defaultValue={this.props.rate}/></span>;
        return (
            <Comment actions={actions} author={info} avatar={this.props.avatar} content={content}
                     datetime={this.props.datetime}/>
        );
    };

}

class ReviewList extends Component {
    render() {
        const data = [
            {
                author: "欧拉王",
                filmName: "西虹市首富",
                rate: 2,
                avatar: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png',
                content: (
                    <p>……开心麻花片子的老问题，<span className="emphatic">剧情分裂</span>，建立一个设定猛耍宝……</p>
                ),
                fullContent: (
                    <p>开心麻花片子的老问题，<span className="emphatic">剧情分裂</span>，建立一个设定猛耍宝，很多耍宝很
                        low很常用，没有新意。为了好玩好笑，剧情发展差点也无所谓了。连情节中最基本的时间概念也没考虑使用，做做文章。</p>
                ),
                datetime: (
                    <Tooltip title={moment().subtract(1, 'days').format('YYYY-MM-DD HH:mm:ss')}>
                        <span>{moment().subtract(1, 'days').fromNow()}</span>
                    </Tooltip>
                ),
            },
            {
                author: "欧拉王",
                filmName: '爱情公寓',
                rate: 1,
                avatar: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png',
                content: (
                    <p>……突然煽情，<span className="emphatic">剧情发展牵强</span>。特效不好看但是感觉挺费钱的……</p>
                ),
                fullContent: (
                    <p>
                        张起灵为了激活铠甲留了一桶血晕了过去，不过马上又复活了，陈美嘉问张起灵为什么又满血复活了，张起灵说了一句让我怀疑人生话:谢谢你的乌鸡白凤丸。我真是r你马了！就当你为了搞笑故意为之吧。但是也太扯了。突然煽情，<span
                        className="emphatic">剧情发展牵强</span>。特效不好看但是感觉挺费钱的，花那么多钱拍个这种垃圾电影。我在看电影之前是反感王传君的，觉得也是多年好朋友，也是自己事业的起点，怎么样都不应该抛弃。现在我明白王传君“演员要有羞耻心”这句话的意思了。领教了。
                    </p>
                ),
                datetime: (
                    <Tooltip title={moment().subtract(2, 'days').format('YYYY-MM-DD HH:mm:ss')}>
                        <span>{moment().subtract(2, 'days').fromNow()}</span>
                    </Tooltip>
                ),
            },
        ];
        return (
            <List
                className="comment-list"
                header={(<span className="byline"><strong>`部分相关评论`</strong></span>)}
                itemLayout="horizontal"
                dataSource={data}
                renderItem={item => (
                    <SingleComment
                        author={item.author}
                        filmName={item.filmName}
                        rate={item.rate}
                        avatar={item.avatar}
                        content={item.content}
                        fullContent={item.fullContent}
                        datetime={item.datetime}
                    />
                )}
            />
        )
    }
}

export default ReviewList