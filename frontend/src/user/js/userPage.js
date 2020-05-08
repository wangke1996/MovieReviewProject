import React, {Component} from 'react';
import {Row, Col, Input, Typography, Select, Spin, Button, Avatar, Modal} from 'antd';
import {ExclamationCircleOutlined} from '@ant-design/icons';
import debounce from 'lodash/debounce';
import {checkUserState, getActiveUsers, searchUser} from "../../libs/getJsonData";
import {image_url} from "../../libs/toolFunctions";
import {UserProfile} from "./userProfile";
import {Uploader} from "./uploader";
import {SentimentProfile} from "../../movieProfile/js/sentimentProfile";
import {RecommendMovie} from "./recommendMovie";

const {confirm} = Modal;
const {Title, Text, Paragraph} = Typography;
const {Search} = Input;
const {Option} = Select;

class UserRemoteSelect extends Component {
    state = {
        data: [],
        value: [],
        fetching: false,
        activeUsers: []
    };

    constructor(props) {
        super(props);
        this.lastFetchId = 0;
        this.fetchUser = debounce(this.fetchUser, 800);
    }

    componentDidMount() {
        this.fetchActiveUsers()
    }

    fetchActiveUsers = () => {
        getActiveUsers(activeUsers => this.setState({activeUsers}));
    };
    activeUserList = () => {
        const {activeUsers} = this.state;
        return activeUsers.map(user => <Button key={user.id} type='link' className='margin-top margin-right'
                                               onClick={() => this.setState({value: {key: user.id, label: user.name}})}>
            <Avatar size='large' src={user.avatar}/>
            {user.name}
        </Button>)
    };
    fetchUser = value => {
        if (!value)
            return;
        this.lastFetchId += 1;
        const fetchId = this.lastFetchId;
        this.setState({data: [], fetching: true});
        searchUser(value, (data) => {
            if (fetchId !== this.lastFetchId) {
                // for fetch callback order
                return;
            }
            this.setState({data, fetching: false});
        });
    };

    handleChange = val => {
        const value = !val ? [] : val[val.length - 1];
        this.setState({
            value,
            data: [],
            fetching: false,
        });
    };

    render() {
        const {fetching, data, value} = this.state;
        return (
            <div>
                <Row type='flex'>
                    <Col span={16}>
                        <Select
                            mode='tags'
                            labelInValue
                            value={value}
                            placeholder="请输入豆瓣用户名或用户ID"
                            notFoundContent={fetching ? <Spin size="small"/> : null}
                            filterOption={false}
                            onSearch={this.fetchUser}
                            onChange={this.handleChange}
                            style={{width: '100%'}}
                        >
                            {data.map(d => (
                                <Option key={d.id}>{d.name}</Option>
                            ))}
                        </Select>
                    </Col>
                    <Col span={4}>
                        <Button onClick={() => this.props.submit(value.key)}>查看用户画像</Button>
                    </Col>
                </Row>
                <Title className='margin-top' level={3}>活跃用户top10</Title>
                <Paragraph type='secondary'>可直接点击选中</Paragraph>
                {this.activeUserList()}
            </div>
        );
    }
}

export class UserPage extends Component {
    state = {
        uid: undefined,
        type: 'user',
        loading: false,
    };
    changeUid = (uid) => {
        checkUserState(uid, res => {
            const state = res.response;
            if (state === 'ok')
                this.setState({uid, type: 'user'});
            else if (state === 'uncached') {
                confirm({
                    title: '数据库中没有缓存该用户的画像，分析需要较长时间，是否换一个用户？',
                    icon: <ExclamationCircleOutlined/>,
                    content: '爬取用户信息并进行分析需要1-10分钟',
                    okText: '好的，我换一个',
                    okType: 'primary',
                    cancelText: '不，我就要看TA',
                    cancelType: 'danger',
                    onOk: () => {
                    },
                    onCancel: () => this.setState({uid, type: 'user'}),
                });
            } else {
                Modal.error({
                    title: '该用户不存在！',
                    content: '目前仅支持豆瓣用户，试试推荐的活跃用户吧！或者上传用户的评论获取兴趣画像~',
                });
            }
        })
    };
    setProfile = (uid) => this.setState({uid, type: 'cache'});

    render() {
        const {uid, type} = this.state;
        return (
            <div id="Content" className="UserProfile">
                <div id="banner">
                    <h2>Hi! 欢迎使用 <strong>用户画像</strong>功能.</h2>
                    <span className="byline">
                        上传用户的影评，挖掘用户性格特点和电影审美
                    </span>
                    <hr/>
                </div>
                <Row type='flex' justify="space-around">
                    <Col span={8}>
                        <Title level={2}>完整用户画像</Title>
                        <Text type='secondary'>仅限豆瓣用户</Text>
                        <UserRemoteSelect submit={uid => this.changeUid(uid)} submitHint='查看用户画像'/>
                    </Col>
                    <Col span={8}>
                        <Title level={2}>用户评论解析</Title>
                        <Text type='secondary'>上传任意用户评论，了解用户兴趣</Text>
                        <Uploader setProfile={(uid) => this.setProfile(uid)}/>
                    </Col>
                </Row>
                {type === 'user' ? <UserProfile uid={uid}/> :
                    <SentimentProfile id={uid} type={type}/>}
                <RecommendMovie id={uid} type={type}/>
            </div>
        )
    }
}