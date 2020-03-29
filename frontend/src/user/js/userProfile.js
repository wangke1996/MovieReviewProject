import React, {Component} from 'react';
import {Tag, Affix, Avatar, Tooltip, Typography} from 'antd'
import RateDistribution from './rateDistribution'
import FavoriteType from './favoriteType'
import ReviewList from './reviewList'
import AgeDistribution from './ageDistribution'
import FavoriteActorInfo from './favoriteActorInfo'
import TotalNum from './totalNum'
import '../css/userProfile.css';
import {getUserProfile} from "../../libs/getJsonData";
import LoadingSpin from "../../common/js/loadingSpin";
import {SentimentProfile} from "../../movieProfile/js/sentimentProfile";

const {Title} = Typography;
const colors = ["magenta", "red", "volcano", "orange", "gold", "lime"];
const tagJumpIds = ["totalNum", "rateDistribution", "favoriteType", "reviewList", "ageDistribution", "favoriteActor"];

export class UserProfile extends Component {
    state = {
        'info': {
            'name': '六金莱',
            'avatar': '/source/images/avatar/SunWuKong.jpg',
            'totalNum': 1234,
            'homePage': 'https://baike.baidu.com/item/六小龄童/142561'
        },
        'tags': ['阅片无数', '真的很严格', '科幻迷', '剧情吐槽狂', '怀旧', '章金莱'],
        'texts': ['阅片无数', '真的很严格', '资深科幻迷', '在涉及电影剧情的40条评价中，负面评价多达80%', '看了834部上个世纪的电影', '可以说是铁杆粉丝了'],
        'distribution': {
            'averageScore': 3,
            'rateDistribution': [{'rate': "1 star", 'reviewNums': 264}, {'rate': "2 star", 'reviewNums': 180},
                {'rate': "3 star", 'reviewNums': 460}, {'rate': "4 star", 'reviewNums': 210},
                {'rate': "5 star", 'reviewNums': 120}],
            'typeDistribution': [{'type': "爱情", 'num': 90}, {'type': "喜剧", 'num': 40},
                {'type': "动作", 'num': 120},
                {'type': "科幻", 'num': 426}, {'type': "纪实", 'num': 30},
                {'type': "艺术", 'num': 0},
                {'type': "恐怖", 'num': 120}, {'type': "剧情", 'num': 150},
                {'type': "冒险", 'num': 100},
                {'type': "动画", 'num': 50}, {'type': "战争", 'num': 108}],
            'ageDistribution': [...Array(60).keys()].map(i => {
                return {year: i + 160, count: Math.round(40 * Math.random())}
            }),
        },

        'favorite': {
            'favoriteActor': {
                'id': 1274392, 'name': '章金莱', 'saw': 5,
                'url': 'https://movie.douban.com/celebrity/1274392/',
                'img': 'https://img1.doubanio.com/view/celebrity/s_ratio_celebrity/public/p1453940528.49.webp',
                'description': '六小龄童本名章金莱，是南猴王“六龄童”章宗义的小儿子。1959年4月12日出生于上海，祖籍浙江绍兴，现为中央电视台、中国电视剧制作中心演员剧团国家一级演员。他出生于“章氏猴戏”世家，从小随父学艺。1976年6月在上海高中毕业后，考入浙江省昆剧团艺校，专攻武生，曾主演昆剧《孙悟空三借芭蕉扇》、《美猴王大闹龙宫》、《武松打店》、《三岔口》、《挑滑车》、《战马超》等，颇受观众好评。他在央视电视剧《西游记》中扮演孙悟空，该剧在美国、日本、德国、法国及东南亚各国播出后，受到广泛好评，六小龄童从此家喻户晓、蜚声中外。'
            }
        },
        'sentiment': {
            'reviewList': [
                {
                    'target': '剧情', 'description': '牵强', 'targetIndex': 33, 'descriptionIndex': 37, 'movie': '七龙珠',
                    'rate': 1, 'date': '2020-02-02 23:33:33',
                    'content': '我们不要一味跟着某一些国家后面去追他们的那种风格，什么《七龙珠》，剧情发展牵强，孙悟空都弄得髭毛乍鬼的，这个不是我们民族的东西！'
                },
                {
                    'target': '剧情', 'description': '混乱', 'targetIndex': 28, 'descriptionIndex': 30,
                    'movie': '大梦西游',
                    'rate': 1, 'date': '2019-12-23 23:33:33',
                    'content': '现在改编的这些电影，完全不尊重原著，是非颠倒，人妖不分，剧情混乱，居然还有孙悟空和白骨精谈恋爱的情节，以至于总有小朋友问我：“六爷爷，孙悟空到底有几个女朋友啊？”'
                },
                {
                    'target': '剧情', 'description': '烂', 'targetIndex': 0, 'descriptionIndex': 3, 'movie': '西游记女儿国',
                    'rate': 1, 'date': '2019-01-23 23:33:33', 'content': '剧情太烂了！戏说不是胡说，改编不是乱编，你们这样是要向全国人民谢罪的！'
                },],
            'worstTarget': '剧情',
            'negativeRate': 0.8,
            'negativeNum': 32,
        },
        flags: Array(6).fill(""),
        height: 0,
        loading: false
    };
    fetchData = () => {
        const {uid} = this.props;
        this.setState({loading: true});
        getUserProfile(uid, data => {
            this.setState(Object.assign({}, data, {loading: false}));
        })
    };

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (this.props.uid !== prevProps.uid)
            this.fetchData()
    }

    componentDidMount() {
        this.updateWindowDimensions();
        window.addEventListener('resize', this.updateWindowDimensions);
        this.fetchData();
    }

    componentWillUnmount() {
        window.removeEventListener('resize', this.updateWindowDimensions);
    }

    updateWindowDimensions = () => {
        this.setState({height: window.innerHeight, flags: this.state.flags});
    };

    handleTagClick = i => {
        const flags = new Array(6).fill("");
        flags[i] = "focused";
        this.setState({flags});
        setTimeout(() => this.setState({flags: Array(6).fill("")}), 3000)
    };

    renderTags = (i, text) => <Tag key={i} color={colors[i]}><a href={"#" + tagJumpIds[i]}
                                                                onClick={() => this.handleTagClick(i)}
                                                                className="scrolly">{text}</a></Tag>;

    render() {
        const {height, loading, flags, info, tags, texts, distribution, favorite, sentiment} = this.state;
        return (
            loading ? <LoadingSpin/> :
                <div className="wrapper style1 align-center">
                    <section id="UserInfo">
                        <header>
                            <Title level={1} type='secondary'>{info.name} 的用户画像</Title>
                            <Affix offsetTop={height / 2}>
                                <Tooltip title='单击头像前往Ta的豆瓣主页'>
                                    <a href={info.homePage}>
                                        <Avatar src={info.avatar} size="large"/>
                                    </a>
                                </Tooltip>
                                <div>
                                    {tags.map((d, i) => this.renderTags(i, d))}
                                </div>
                            </Affix>
                        </header>
                    </section>
                    <section id="AbstractGraphs">
                        <div className="AbstractGraph">
                            <div className="row">
                                <TotalNum flag={flags[0]} text={texts[0]} totalNum={info.totalNum}/>
                                <RateDistribution flag={flags[1]} averageScore={distribution.averageScore}
                                                  text={texts[1]}
                                                  data={distribution.rateDistribution}/>
                            </div>
                            <div className="row">
                                <FavoriteType flag={flags[2]} data={distribution.typeDistribution} text={texts[2]}/>
                                <ReviewList flag={flags[3]} name={info.name} avatar={info.avatar}
                                            reviewList={sentiment.reviewList}
                                            worstTarget={sentiment.worstTarget}
                                            negativeRate={sentiment.negativeRate}
                                            negativeNum={sentiment.negativeNum} text={texts[3]}/>
                            </div>
                            <div className="row">
                                <AgeDistribution flag={flags[4]} data={distribution.ageDistribution} tag={tags[4]}
                                                 text={texts[4]}/>
                                <FavoriteActorInfo flag={flags[5]} name={favorite.favoriteActor.name} text={texts[5]}
                                                   img={favorite.favoriteActor.img} url={favorite.favoriteActor.url}
                                                   saw={favorite.favoriteActor.saw}
                                                   description={favorite.favoriteActor.description}/>
                            </div>
                        </div>
                    </section>
                    <SentimentProfile id={this.props.uid} type='user'/>
                </div>

        )
    }
}

UserProfile.defaultProps = {
    uid: 'perhapsfish'
};
