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
    constructor(props) {
        super(props);
        this.props.flag = "";
    }

    render() {
        let avatar_src = process.env.PUBLIC_URL + "/source/images/avatar/SunWuKong.jpg";
        const data = [
            {
                author: "六金莱",
                filmName: "大梦西游",
                rate: 1,
                avatar: avatar_src,
                content: (
                    <p>……是非颠倒，人妖不分，<span className="emphatic">剧情混乱</span>……</p>
                ),
                fullContent: (
                    <p>现在改编的这些电影，完全不尊重原著，是非颠倒，人妖不分，<span className="emphatic">剧情混乱</span>，居然还有孙悟空和白骨精谈恋爱的情节，以至于总有小朋友问我：“六爷爷，孙悟空到底有几个女朋友啊？”
                    </p>
                ),
                datetime: (
                    <Tooltip title={moment().subtract(1, 'days').format('YYYY-MM-DD HH:mm:ss')}>
                        <span>{moment().subtract(1, 'days').fromNow()}</span>
                    </Tooltip>
                ),
            },
            {
                author: "六金莱",
                filmName: '七龙珠',
                rate: 1,
                avatar: avatar_src,
                content: (
                    <p>……<span className="emphatic">剧情发展牵强</span>，孙悟空都弄得髭毛乍鬼的……</p>
                ),
                fullContent: (
                    <p>
                        我们不要一味跟着某一些国家后面去追他们的那种风格，什么《七龙珠》，<span className="emphatic">剧情发展牵强</span>，孙悟空都弄得髭毛乍鬼的，这个不是我们民族的东西！
                    </p>
                ),
                datetime: (
                    <Tooltip title={moment().subtract(2, 'days').format('YYYY-MM-DD HH:mm:ss')}>
                        <span>{moment().subtract(2, 'days').fromNow()}</span>
                    </Tooltip>
                ),
            },
            {
                author: "六金莱",
                filmName: '西游记女儿国',
                rate: 1,
                avatar: avatar_src,
                content: (
                    <p>……<span className="emphatic">剧情太烂了</span>！戏说不是胡说……</p>
                ),
                fullContent: (
                    <p>
                        <span className="emphatic">剧情太烂了</span>！戏说不是胡说，改编不是乱编，你们这样是要向全国人民谢罪的！
                    </p>
                ),
                datetime: (
                    <Tooltip title={moment().subtract(6, 'seconds').format('YYYY-MM-DD HH:mm:ss')}>
                        <span>{moment().subtract(6, 'seconds').fromNow()}</span>
                    </Tooltip>
                ),
            },
        ];
        return (
            <div className={"6u " + this.props.flag} id="reviewList">
                <header>
                    <h2>对<span className="emphatic">剧情</span>最为挑剔</h2>
                    <span className="byline">对电影{"剧情"}的评价中，负面评价多达<span className="emphatic">80%</span></span>
                </header>
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
            </div>
        )
    }
}

export default ReviewList