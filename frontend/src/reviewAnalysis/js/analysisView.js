import React, {Component} from 'react'
import {Input, Button, message, Table, Typography, Row, Col, Rate} from "antd";
import {analysisReview, getDemo} from "../../libs/getJsonData";
import '../css/analysisView.css'
import {Uploader} from "../../user/js/uploader";

const {Text, Title} = Typography;
const {TextArea} = Input;
const sentimentText = {POS: '积极', NEU: '中性', NEG: '消极'};
const columns = [
    {
        title: '属性',
        dataIndex: 'target',
        key: 'target',
        // render: text => <a>{text}</a>,
    },
    {
        title: '描述',
        dataIndex: 'description',
        key: 'description',
    },
    {
        title: '情感倾向',
        dataIndex: 'sentiment',
        key: 'sentiment',
        render: sent => <Text strong>{sentimentText[sent]}</Text>
    },
    {
        title: '评论片段',
        key: 'sentence',
        dataIndex: 'sentence',
        render: (sentence, record) => {
            const {target, description} = record;
            let index1 = sentence.indexOf(target), index2 = sentence.indexOf(description);
            let len1 = target.length, len2 = description.length;
            if (index1 > index2) {
                let tmp = index2;
                index2 = index1;
                index1 = tmp;
                tmp = len2;
                len2 = len1;
                len1 = tmp;
            }
            return <Text>
                {sentence.slice(0, index1)}
                <Text mark>{sentence.slice(index1, index1 + len1)}</Text>
                {sentence.slice(index1 + len1, index2)}
                <Text mark>{sentence.slice(index2, index2 + len2)}</Text>
                {sentence.slice(index2 + len2)}
            </Text>
        }
    }
];

export class AnalysisView extends Component {
    state = {
        text: undefined,
        result: undefined,
        score: undefined,
        loading: false,
    };
    onChange = ({target: {value}}) => {
        this.setState({text: value});
    };
    getDemo = () => {
        getDemo((text) => this.setState({text}))
    };
    onSubmit = () => {
        const {text} = this.state;
        if (!text) {
            message.error('文本内容为空！');
            return;
        }
        this.setState({loading: true});
        const info = message.loading({content: '正在解析，请稍候……', key: 'msg'}, 0)
        analysisReview(text, (res) => {
            setTimeout(info, 1);
            message.success({content: '解析成功！', key: 'msg'}, 2);
            this.setState({loading: false, result: res.data, score: res.score})
        });
    };
    resultTable = () => {
        const {result} = this.state;
        if (!result)
            return [];
        const data = result.map((d, i) => Object.assign(d, {key: i}));
        return <Table id='analysisTable' columns={columns}
                      dataSource={data} /*rowClassName={record => record.sentiment}*//>
    };
    predictScore = () => {
        const {score} = this.state;
        if (!score)
            return [];
        return <div id='predictScore'>
            <h2 className='center margin-bottom'>情感强度预测</h2>
            <Rate disabled defaultValue={score}/>
        </div>
    };

    render() {
        const {text, loading} = this.state;
        return (
            <div>
                <div id="Content" className="UserProfile">
                    <div id="banner">
                        <h2>Hi! 欢迎使用 <strong>评论解析</strong>功能.</h2>
                        <span className="byline">
                        上传用户影评，挖掘细粒度观点
                    </span>
                        <hr/>
                    </div>
                </div>
                <Row justify="space-around" type='flex'>
                    <Col span={12}>
                        <Title level={2}>单条评论解析</Title>
                        <TextArea
                            id='reviewInput'
                            size='large'
                            value={text}
                            onChange={this.onChange}
                            placeholder="请输入一条电影评论文本"
                            autoSize={{minRows: 5, maxRows: 10}}
                        />
                        <div className='margin-top margin-bottom'>
                            <Button className='margin-right' size='large' type='danger' loading={loading}
                                    onClick={this.onSubmit}>解析 GO</Button>
                            <Button className='margin-right' size='large' type='default'
                                    onClick={this.getDemo}>查看样例</Button>
                            <Button size='large' type='dashed'
                                    onClick={() => this.setState({text: undefined})}>清空输入</Button>
                        </div>
                        {this.resultTable()}
                        {this.predictScore()}
                    </Col>
                    <Col span={8}>
                        <Title level={2}>批量评论解析</Title>
                        <Text type='secondary'>上传评论文本文件，获取细粒度观点分析结果</Text>
                        <Uploader setProfile={()=>{}}/>
                    </Col>
                </Row>
            </div>
        )
    }
}