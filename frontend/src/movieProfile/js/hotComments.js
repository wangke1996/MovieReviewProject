import React, {Component} from 'react';
import {getMovieComments} from '../../libs/getJsonData'
import {Comment, Icon, List, Rate, Tabs, Tooltip} from "antd";
import moment from "moment";
import {image_url} from "../../libs/toolFunctions";
import '../css/hotComments.css'
import LoadingSpin from '../../common/js/loadingSpin'

class SingleComment extends Component {
    state = {
        full: false,
    };
    action = () => this.setState({full: !(this.state.full)});

    render() {
        const full = this.state.full;
        const actions = [<span className="usefulCount">
            <Tooltip title={this.props.likes + '人赞成这条评论'}>
                <Icon
                    type="like"
                    theme='filled'
                />
                <span className='number'> {this.props.likes} </span>
            </Tooltip>
            </span>];
        const longComment = this.props.content !== this.props.fullContent;
        if (longComment)
            actions.push(<a onClick={this.action}>{full ? "简略信息" : "查看全文"}</a>);
        const content = full || !longComment ? this.props.fullContent : this.props.content + '……';
        const info = <span>作者：{this.props.author} | 打分：<Rate disabled defaultValue={this.props.rate}/></span>;
        return (
            <Comment actions={actions} author={info} avatar={this.props.avatar} content={content}
                     datetime={this.props.datetime}/>
        );
    };

}

class CommentsList extends Component {

    render() {
        const data = [];
        this.props.commentsData.slice(0, this.props.maxReviewNum).forEach((d, i) => {
            data[i] = {
                rate: d['rating']['value'],
                fullContent: d['content'],
                content: d['content'].slice(0, this.props.maxReviewLen),
                author: (<a href={d['author']['alt']}>{d['author']['name']}</a>),
                avatar: image_url(d['author']['avatar']),
                dateTime: (<Tooltip title={d['created_at']}>
                    <span>{moment(d['created_at'], 'YYYY-MM-DD HH:mm:ss').fromNow()}</span>
                </Tooltip>),
                likes: d['useful_count']
            };
        });
        return (
            <List
                className="comment-list"
                header={this.props.title}//{(<span className="byline"><strong>{this.props.title}</strong></span>)}
                itemLayout="horizontal"
                dataSource={data}
                renderItem={item => (
                    <SingleComment
                        author={item.author}
                        rate={item.rate}
                        avatar={item.avatar}
                        content={item.content}
                        fullContent={item.fullContent}
                        datetime={item.dateTime}
                        likes={item.likes}
                    />
                )}
            />
        )
    }
}

CommentsList.defaultProps = {
    maxReviewNum: 5,
    maxReviewLen: 100,
    commentsData: [],
    title: '热评精选'
};

class HotComments extends Component {
    loadData(movieID) {
        getMovieComments(movieID, (data) => {
            this.setState({
                commentsData: data,
                loadedFlag: true
            })
        });
    }

    constructor(props) {
        super(props);
        this.state = {
            commentsData: [],
            loadedFlag: false,
        };
        this.loadData(this.props.movieID);
    }

    render() {
        if (!this.state.loadedFlag)
            return (<LoadingSpin/>);
        const TabPane = Tabs.TabPane;
        const tabsData = [[], [], [], [], []];
        this.state.commentsData.forEach(d => {
            let rate = parseInt(d['rating']['value']);
            rate = rate > 5 ? 5 : rate < 1 ? 1 : rate;
            d['rating']['value'] = rate;
            tabsData[rate - 1].push(d);
        });
        const tabPanes = [];
        tabsData.forEach((d, i) => {
            tabPanes[i] = (
                <TabPane tab={<span><Rate disabled defaultValue={i + 1} style={{fontSize: '0.5em'}}/></span>}
                         key={(i + 1).toString()}>
                    <CommentsList commentsData={d} title={(i + 1) + '星短评精选'}/>
                </TabPane>);
        });
        return (
            <Tabs defaultActiveKey="1">
                {tabPanes}
            </Tabs>
        )
    }
}

export default HotComments;