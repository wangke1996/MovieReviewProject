import React, {Component} from 'react'
import {Comment, Tooltip, List, Rate, Button, Typography, Card} from 'antd';
import moment from 'moment';
import {image_url, toPercent} from "../../libs/toolFunctions";

const {Text, Paragraph} = Typography;
const {Meta} = Card;

class SingleComment extends Component {
    state = {
        full: false,
    };
    action = () => this.setState({full: !(this.state.full)});

    render() {
        const full = this.state.full;
        const actions = [<Button type='link' onClick={this.action}>{full ? "简略信息" : "查看全文"}</Button>];
        const content = full ? this.props.fullContent : this.props.content;
        const info = <span>作者：{this.props.author} | 影片：<strong>{this.props.movie}</strong> | 打分：<Rate disabled
                                                                                                      defaultValue={this.props.rate}/></span>;
        return (
            <Comment actions={actions} author={info} avatar={this.props.avatar} content={content}
                     datetime={this.props.datetime}/>
        );
    };

}

class ReviewList extends Component {
    transformData = () => {
        const {reviewList, name, avatar} = this.props;
        return reviewList.map(d => {
            const {movie, rate, content, target, description, targetIndex, descriptionIndex, date, url, movie_img} = d;
            const start = Math.max(Math.min(targetIndex, descriptionIndex) - 10, 0);
            const end = Math.min(Math.max(targetIndex + target.length, descriptionIndex + description.length) + 10, content.length);
            const partContent = content.slice(start, end);
            let index1 = targetIndex, index2 = descriptionIndex, len1 = target.length, len2 = description.length;
            if (index1 > index2) {
                let tmp = index2;
                index2 = index1;
                index1 = tmp;
                tmp = len2;
                len2 = len1;
                len1 = tmp;
            }
            const partIndex1 = index1 - start, partIndex2 = index2 - start;
            const prefix = start === 0 ? '' : '……';
            const postfix = end === content.length ? '' : '……';
            return {
                author: name, avatar, rate,
                movie: <Tooltip title={
                    <a href={url}>
                        <Card hoverable style={{width: 240}} cover={<img alt={movie} src={image_url(movie_img)}/>}>
                            <Meta title={movie} description='点击前往电影详情页'/>
                        </Card>
                    </a>
                }>
                    {movie}
                </Tooltip>,
                datetime:
                    <Tooltip title={date}>
                        <span>{moment(date, 'YYYY-MM-DD HH:mm:ss').fromNow()}</span>
                    </Tooltip>,
                content: <Paragraph>
                    {prefix}
                    {partContent.slice(0, partIndex1)}
                    <Text mark>{partContent.slice(partIndex1, partIndex1 + len1)}</Text>
                    {partContent.slice(partIndex1 + len1, partIndex2)}
                    <Text mark>{partContent.slice(partIndex2, partIndex2 + len2)}</Text>
                    {partContent.slice(partIndex2 + len2)}
                    {postfix}
                </Paragraph>,
                fullContent: <Paragraph style={{textAlign: 'left'}}>
                    {content.slice(0, index1)}
                    <Text mark>{content.slice(index1, index1 + len1)}</Text>
                    {content.slice(index1 + len1, index2)}
                    <Text mark>{content.slice(index2, index2 + len2)}</Text>
                    {content.slice(index2 + len2)}
                </Paragraph>
            }
        })
    };

    render() {
        const {worstTarget, negativeRate, negativeNum, text, flag} = this.props;
        if (negativeNum === 0) {
            return <div className={"6u " + flag} id="reviewList">
                <header>
                    <h2>不挑剔</h2>
                    {/*<span className="byline">对电影{worstTarget}的评价中，负面评价多达<span*/}
                    {/*    className="emphatic">{toPercent(negativeRate)}</span></span>*/}
                    <span className='byline'>{text}</span>
                </header>
            </div>
        }
        return (
            <div className={"6u " + flag} id="reviewList">
                <header>
                    <h2>对<span className="emphatic">{worstTarget}</span>最为挑剔</h2>
                    {/*<span className="byline">对电影{worstTarget}的评价中，负面评价多达<span*/}
                    {/*    className="emphatic">{toPercent(negativeRate)}</span></span>*/}
                    <span className='byline'>{text}</span>
                </header>
                <List
                    className="comment-list"
                    header={(<span className="byline"><strong>部分相关评论</strong></span>)}
                    itemLayout="horizontal"
                    dataSource={this.transformData()}
                    renderItem={item => (
                        <SingleComment
                            author={item.author}
                            movie={item.movie}
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