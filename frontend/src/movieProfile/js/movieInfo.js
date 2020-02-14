import React, {Component} from 'react';
import {Rate, Popover, Card, Comment, Tooltip, List, Icon, Tabs, Divider, Carousel, Tag, message} from 'antd';
import {Chart, Geom, Axis, Coord, Label, Tooltip as BizTooltip} from 'bizcharts';
import {getMovieInfo, getMovieComments, getMoviePhotos} from '../../libs/getJsonData'
import moment from 'moment'
import '../css/movieInfo.css'



class CarouselImage extends Component {
    render() {
        const images = [];
        this.props.datas.forEach(d => images.push(<img alt="" src={image_url(d)}/>));
        return (
            <Carousel effect='fade' autoplay>
                {images}
            </Carousel>
        )
    }
}

class PhotoGrid extends Component {
    render() {
        let i, j;
        let data = [];
        let totalPhotoNum = this.props.photos.length;
        if (totalPhotoNum === 0)
            return (<div/>);
        let num = this.props.carouselNum;
        for (i = 0; i < 9; i++) {
            data[i] = [];
            for (j = 0; j < num; j++) {
                let index = (i * num + j) % totalPhotoNum;
                data[i].push(this.props.photos[index]['cover']);
            }
        }

        return (
            <div>
                <List
                    grid={{gutter: 10, column: 2}}
                    dataSource={data.slice(0, 2)}
                    renderItem={item => (
                        <List.Item>
                            <CarouselImage datas={item}/>
                        </List.Item>
                    )}
                />

                <List
                    grid={{gutter: 10, column: 3}}
                    dataSource={data.slice(2, 5)}
                    renderItem={item => (
                        <List.Item>
                            <CarouselImage datas={item}/>
                        </List.Item>
                    )}
                />

                <List
                    grid={{gutter: 10, column: 4}}
                    dataSource={data.slice(5, 9)}
                    renderItem={item => (
                        <List.Item>
                            <CarouselImage datas={item}/>
                        </List.Item>
                    )}
                />
            </div>
        )

    }
}

PhotoGrid.defaultProps = {
    carouselNum: 6
};

class PeopleAvatar extends Component {
    renderPeople(data) {
        const {Meta} = Card;
        const content = (
            <Card hoverable cover={<img alt="" src={image_url(data['avatars']['small'])}/>}>
                <Meta title={data['name']} description={(<a href={data['alt']}>点此查看{data['name']}的详细信息</a>)}/>
            </Card>
        );
        return (
            <Popover content={content}>
                <a href={data['alt']}>{data['name']} </a>
            </Popover>
        )
    }

    render() {
        const people = [];
        let i;
        for (i in this.props.datas)
            people.push(this.renderPeople(this.props.datas[i]));
        return (
            <span>
            {people}
            </span>
        )
    }
}

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
                <span style={{paddingLeft: 8, cursor: 'auto'}}> {this.props.likes} </span>
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

class HotComments extends Component {

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

HotComments.defaultProps = {
    maxReviewNum: 5,
    maxReviewLen: 100,
    commentsData: [],
    title: '热评精选'
};

class HotCommentsTabs extends Component {
    render() {
        const TabPane = Tabs.TabPane;
        const tabsData = [[], [], [], [], []];
        this.props.commentsData.forEach(d => {
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
                    <HotComments commentsData={d} title={(i + 1) + '星短评精选'}/>
                </TabPane>);
        });
        return (
            <Tabs defaultActiveKey="1">
                {tabPanes}
            </Tabs>
        )
    }
}

class MovieTags extends Component {
    render() {
        const presetColors = ['magenta', 'red', 'volcano', 'orange', 'gold', 'lime', 'green', 'cyan', 'blue', 'geekblue', 'purple']
        const tagElements = [];
        this.props.tags.forEach((d, i) => {
            let color = presetColors[i % presetColors.length];
            tagElements.push(<Tag color={color}>{d}</Tag>);
        });
        return (
            <span className="MovieTags">
                {tagElements}
            </span>
        );
    }
}

class MovieInfo extends Component {
    loadJsons(movieID) {
        this.setState((state) => {
            state.info = {};
            state.comments = [];
            state.photos = [];
            state.reviewsTrend = [];
            state.loadedFlag = [0, 0, 0, 0];
            return state;
        });
        getMovieInfo(movieID, (info) => {
            this.setState((state) => {
                state.info = info;
                state.loadedFlag[0] += 1;
                return state;
            })
        });
        getMovieComments(movieID, (comments) => {
            this.setState((state) => {
                state.comments = comments;
                state.loadedFlag[1] += 1;
                return state;
            })
        });
        getMoviePhotos(movieID, (photos) => {
            this.setState((state) => {
                state.photos = photos;
                state.loadedFlag[2] += 1;
                return state;
            })
        });
    }

    constructor(props) {
        super(props);
        this.state = {
            info: {},
            comments: [],
            photos: [],
            loadedFlag: [0, 0, 0],
            useDefault: false
        };
        this.loadJsons(this.props.movieID);
    }

    render() {

        const data = this.state.info;
        const hot_comments = this.state.comments;
        const photos = this.state.photos;

        function get_rate_data() {
            let rate_data = [];
            let rate;
            let i = 0;
            let total = 0;
            for (rate in data['rating']['details'])
                total += data['rating']['details'][rate];
            data['ratings_count'] = total;
            for (rate in data['rating']['details']) {
                rate_data[i] = {
                    'rate': rate + '星',
                    'num': data['rating']['details'][rate],
                    'percent': data['rating']['details'][rate] / total
                };
                i++;
            }
            return rate_data;
        }

        if (this.state.loadedFlag.indexOf(0) !== -1)
            return (<div/>);
        if (isEmpty(data)) {
            this.setState((state) => {
                state.useDefault = true;
                return state;
            });
            this.loadJsons(this.props.defaultMovieID);
            return (<div/>)
        }
        if (this.state.useDefault) {
            message.info('无法找到指定电影的相关信息，已为您显示最近热门电影“' + data['title'] + '”', 10);
        }
        // console.log('render!', this.state);
        return (
            <div className="container">
                <div className="row">
                    <div className="8u">
                        <header><h2>{data['title']}({data['year']})</h2><MovieTags tags={data['tags']}/></header>
                        <div className="row">
                            <div className="4u">
                                <a href={data['images']['large']} className="image">
                                    <img src={image_url(data['images']['large'])} alt=""/>
                                </a>
                            </div>
                            <div className="8u summary">
                                <header><h3>剧情简介</h3></header>
                                <p>{data['summary']}</p>
                            </div>
                        </div>
                        <hr/>
                        <div className="hotComments">
                            <header><span className="byline"><strong>{data['title'] + '的热门短评'}</strong></span></header>
                            <HotCommentsTabs commentsData={hot_comments}/>
                        </div>
                    </div>
                    <div className="4u">
                        <section>
                            <header>
                                <h2>总体评分<span
                                    className="emphatic">{data['rating']['average']}</span>/{data['rating']['max']}
                                </h2>
                                <span className="byline">
                                    <Rate disabled allowHalf defaultValue={parseInt(data['rating']['stars']) / 10}/>
                                    {data['ratings_count']}人参与评价
                                </span>
                                <Chart height={250} data={get_rate_data()} forceFit>
                                    <Coord transpose/>
                                    <Axis name="rate"/>
                                    <Axis name="num"/>
                                    <BizTooltip/>
                                    <Geom type="interval" position="rate*num">
                                        <Label content='num' offset={0} textStyle={
                                            {
                                                textAlign: 'right',
                                                fill: 'white'
                                            }
                                        }
                                               formatter={(text, item, index) => (100 * item.point['percent']).toFixed(2) + '%'}
                                        />
                                    </Geom>
                                </Chart>
                                <div>
                                    <Divider orientation="left">基本信息</Divider>
                                    <p>导演：<PeopleAvatar datas={data['directors']}/></p>
                                    <p>编剧：<PeopleAvatar datas={data['writers']}/></p>
                                    <p>主演：<PeopleAvatar datas={data['casts']}/></p>
                                    <p>类型：{data['genres'].join('/')}</p>
                                    <p>国家/地区：{data['countries'].join('/')}</p>
                                    <p>上映日期：{data['mainland_pubdate']||data['pubdates'].join('/')}</p>
                                    <p>片长：{data['durations']}</p>
                                    <p>别名：{data['aka'].join('/')}</p>
                                </div>
                                <div>
                                    <Divider>剧照</Divider>
                                    <PhotoGrid photos={photos}/>
                                </div>
                            </header>
                        </section>
                    </div>
                </div>
            </div>
        )
    }
}

MovieInfo.defaultProps = {
    defaultMovieID: '26266893'
};
export default MovieInfo;