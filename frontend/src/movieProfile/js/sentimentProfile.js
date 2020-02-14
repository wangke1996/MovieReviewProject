import React, {Component} from 'react';
import ReactEcharts from 'echarts-for-react';
import {Collapse, Badge, Button, List, Skeleton, Divider, Row, Col, Radio, Typography, Select} from 'antd';
import {
    getRelatedSentences,
    getTargetDetail,
    getTargetFreqs,
    getTargetList,
    searchTarget
} from "../../libs/getJsonData";
import LoadingSpin from "../../common/js/loadingSpin";
import {sum} from "../../libs/toolFunctions";
import '../css/sentimentProfile.css';
import {WordCloud} from "./wordCloud";
import {SearchSelect} from "./searchSelect";

const {Panel} = Collapse;
const {Title, Text, Paragraph} = Typography;
const {Option} = Select;
const colors = {POS: 'lime', NEU: 'orange', NEG: 'red'};
// api
// SentimentPie.props={
//     freqs:{POS:123,NEU:456,NEG:789}
// }
class SentimentPie extends Component {
    getOption = () => {
        const {freqs} = this.props;
        const data = [
            {name: '积极评价', value: freqs.POS, key: 'POS'},
            {name: '中性评价', value: freqs.NEU, key: 'NEU'},
            {name: '消极评价', value: freqs.NEG, key: 'NEG'},];
        return {
            series: [{
                type: 'pie',
                data: data,
                animation: true,
                label: {
                    position: 'outer',
                    alignTo: 'none',
                    bleedMargin: 5
                },
                itemStyle: {
                    color: params => colors[params.data.key],
                }
            }]
        }
    };

    render() {
        return (
            <ReactEcharts
                option={this.getOption()}
                // style={{height: window.innerHeight, width: '80%'}}
            />
        )
    }
}

class ReviewList extends Component {
    state = {
        initLoading: true,
        loading: false,
        data: [],
        list: [],
        startIndex: 0,
        count: 3,
        loadedAll: false,
    };
    getData = () => {
        const {movieID, target, sentiment, description} = this.props;
        const {startIndex, count, data, list} = this.state;
        const query = {
            movieID: movieID,
            target: target,
            sentiment: sentiment,
            description: description,
            startIndex: startIndex,
            count: count
        };
        this.setState({
            loading: true,
            list: list.concat([...new Array(count)].map(() => ({loading: true, text: ''}))),
        });

        getRelatedSentences(query, res => {
            const newData = data.concat(res.data);
            this.setState({
                initLoading: false,
                loading: false,
                loadedAll: res.loadedAll,
                startIndex: res.startIndex,
                data: newData,
                list: newData.map(text => {
                    return {text, loading: false}
                })
            }, () => {
                // Resetting window's offsetTop so as to display react-virtualized demo underfloor.
                // In real scene, you can using public method of react-virtualized:
                // https://stackoverflow.com/questions/46700726/how-to-use-public-method-updateposition-of-react-virtualized
                window.dispatchEvent(new Event('resize'));
            });
        });
    };

    componentDidMount() {
        this.getData();
    }

    render() {
        const {initLoading, loading, loadedAll, list} = this.state;
        const loadMore = loadedAll ? <Divider>已加载全部</Divider> :
            !initLoading && !loading ? (
                <div
                    style={{
                        textAlign: 'center',
                        marginTop: 12,
                        height: 32,
                        lineHeight: '32px',
                    }}
                >
                    <Button onClick={this.getData}>加载更多</Button>
                </div>
            ) : null;

        return (
            <List
                className="loadmore-list"
                loading={initLoading}
                itemLayout="horizontal"
                loadMore={loadMore}
                dataSource={list}
                renderItem={item => (
                    <List.Item>
                        <Skeleton title={false} loading={item.loading} active>
                            {item.text}
                        </Skeleton>
                    </List.Item>
                )}
            />
        );
    }
}

// api
// TargetSentimentTree.props={
//     sentimentTree:{
//         POS:[{name:'好',freq:1000},{name:'棒棒',freq:800}],
//         NEU:[],
//         NEG:[]
//     }
// }
class TargetSentimentTree extends Component {
    generatePanel = (descriptionList, title, sentiment) => <Panel className={sentiment}
                                                                  header={<Badge
                                                                      count={sum(descriptionList.map(d => d.freq))}
                                                                      overflowCount={1000}>{title}</Badge>}
                                                                  key={sentiment}>
        <Collapse accordion>
            {descriptionList.map(d =>
                <Panel key={d.name} header={<Badge count={d.freq}>{d.name}</Badge>}>
                    <ReviewList movieID={this.props.movieID} target={this.props.target} description={d.name}
                                sentiment={sentiment}/>
                </Panel>)}
        </Collapse>
    </Panel>;

    render() {
        const {POS, NEU, NEG} = this.props.sentimentTree;
        return (
            <div id='targetDetail'>
                <Collapse accordion>
                    {this.generatePanel(POS, '正面评价', "POS")}
                    {this.generatePanel(NEU, '中性评价', "NEU")}
                    {this.generatePanel(NEG, '负面评价', "NEG")}
                </Collapse>
            </div>
        )
    }
}


// class Target extends Component {
//     render() {
//         const {name, freq} = this.props;
//         return (
//             <Badge count={freq}>
//                 <Button type='link' onClick={() => this.props.setTarget(name)}>{name}</Button>
//             </Badge>
//         )
//     }
// }

// api
// HotTarget.state.data=[
//     {name:'hot No1',freq:999},
//     {name:'hot No2',freq:888}
// ];
class HotTarget extends Component {
    state = {
        data: [],
        loaded: false,
    };
    getHotTarget = () => {
        const {movieID, sortBy} = this.props;
        const query = {movieID, sortBy};
        getTargetList(query, data => this.setState({data, loaded: true}));
    };

    componentDidMount() {
        this.getHotTarget();
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (prevProps.movieID !== this.props.movieID || prevProps.sortBy !== this.props.sortBy)
            this.getHotTarget();
    }

    render() {
        if (!this.state.loaded)
            return <LoadingSpin/>;
        return (<div className='center'>
            {this.state.data.map((d, i) => <Badge className='margin-right margin-left' key={i} count={d.freq}
                                                  overflowCount={1000}>
                <Button type='link' onClick={() => this.props.setTarget(d.name)}>{d.name}</Button>
            </Badge>)}

        </div>);
    }
}


class TargetDetail extends Component {
    state = {
        loaded: false,
        freqs: {},
        descriptions: [],
        sentimentTree: {}
    };
    setStateFromData = (data) => {
        let freqs = {};
        for (let sentiment in data)
            freqs[sentiment] = sum(data[sentiment].map(d => d.freq));
        const descriptions = Array.prototype.concat(...Object.entries(data).map(([k, v]) => v.map(d => Object.assign({'sentiment': k}, d))));
        const sentimentTree = data;
        this.setState({freqs, descriptions, sentimentTree, loaded: true});
    };
    getData = () => {
        this.setState({loaded: false});
        const {target, movieID} = this.props;
        const query = {target, movieID};
        getTargetDetail(query, this.setStateFromData);
    };

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (this.props.movieID !== prevProps.movieID || this.props.target !== prevProps.target) {
            this.getData();
        }
    }

    render() {
        const {movieID, target} = this.props;
        const {freqs, sentimentTree, descriptions, loaded} = this.state;
        if (!loaded || !target)
            return <LoadingSpin/>;
        return (
            <Row>
                <Col span={8}>
                    <Title className='center' level={4}><Text mark>{target}</Text>相关评价情感分布图</Title>
                    <SentimentPie freqs={freqs}/>
                </Col>
                <Col span={8}>
                    <Title className='center' level={4}><Text mark>{target}</Text>相关描述及评论</Title>
                    <TargetSentimentTree movieID={movieID} target={target} sentimentTree={sentimentTree}/>
                </Col>
                <Col span={8}>
                    <Title className='center' level={4}><Text mark>{target}</Text>相关描述词云</Title>
                    <WordCloud descriptions={descriptions}/>
                </Col>
            </Row>
        )
    }
}

class SearchTarget extends Component {
    queryFunction = (value, callback) => searchTarget(this.props.movieID, value, callback);
    makeOptions = (targets) => targets.map(d => <Option key={d.name} value={d.name}><Text strong>{d.name}</Text><Text
        type='secondary' className='right'>{d.freq}条相关评价</Text></Option>);

    render() {
        return (
            <div className='center margin-bottom'>
                <SearchSelect queryFunction={this.queryFunction.bind(this)}
                              setValue={this.props.setTarget.bind(this)}
                              makeOptions={this.makeOptions}/>
            </div>
        )
    }
}

export class SentimentProfile extends Component {
    state = {
        sortBy: 'freq',
        target: null,
    };
    setTarget = (target) => this.setState({target});
    onChange = e => {
        this.setState({sortBy: e.target.value});
    };

    render() {
        const {movieID} = this.props;
        const {sortBy, target} = this.state;
        return (
            <div>
                <Radio.Group className='center margin-bottom' defaultValue="freq" buttonStyle="solid"
                             onChange={this.onChange}>
                    <Radio.Button value="freq">高频属性</Radio.Button>
                    <Radio.Button value="POS">积极属性</Radio.Button>
                    <Radio.Button value="NEG">消极属性</Radio.Button>
                </Radio.Group>
                <HotTarget movieID={movieID} sortBy={sortBy} setTarget={this.setTarget.bind(this)}/>
                <SearchTarget movieID={movieID} setTarget={this.setTarget.bind(this)}/>
                <TargetDetail movieID={movieID} target={target} setTarget={this.setTarget.bind(this)}/>
            </div>
        )
    }
}