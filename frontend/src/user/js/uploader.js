import React, {Component} from 'react';
import {Upload, message, Spin, Button, Popconfirm,} from 'antd';
import {InboxOutlined} from '@ant-design/icons';
import {analysisUploadedFile, download, wrapUrl} from "../../libs/getJsonData";

const {Dragger} = Upload;


export class Uploader extends Component {
    state = {
        file: '',
        loading: false,
        cacheId: null
    };

    onFileChange = (info) => {
        const {status} = info.file;
        if (status === 'done') {
            message.success(`${info.file.name} 上传成功`);
            this.setState({file: info.file.response})
        } else if (status === 'error') {
            message.error(`${info.file.name} 上传失败`);
        }
    };

    onSubmit = () => {
        this.setState({loading: true, cacheId: null});
        analysisUploadedFile(this.state.file, (res) => {
            if (res.status === 'success') {
                message.success('解析成功！');
                this.setState({cacheId: res.cacheID});
                this.props.setProfile(res.cacheID)
            } else
                message.error(res.message);
            this.setState({loading: false});
        })
    };

    downloadResult = () => {
        const {cacheId} = this.state;
        if (!cacheId) {
            message.error('出错啦，请重新上传');
            return;
        }
        download(cacheId, () => {
        });
    };

    render() {
        const {loading, cacheId} = this.state;
        return (
            <Spin className='loadingSpin margin-bottom' size='large' tip='正在解析上传的评论' spinning={loading}>
                <Dragger name='file' action={wrapUrl('/upload', false)} multiple={false} onChange={this.onFileChange}>
                    <p className="ant-upload-drag-icon">
                        <InboxOutlined/>
                    </p>
                    <p className="ant-upload-text">单击或拖拽上传评论文本</p>
                    <p className="ant-upload-hint">
                        请上传utf8编码的文本文件，若上传多个文件，仅解析最后一次成功上传的文件
                    </p>
                </Dragger>
                <Popconfirm className='large margin-top' title="确认上传？" onConfirm={this.onSubmit}>
                    <Button size='large' type='danger'>确认上传</Button>
                </Popconfirm>
                {!loading && cacheId ? <Button size='large' className='margin-left' type='link' onClick={this.downloadResult}>下载解析结果</Button> : []}
            </Spin>
        )
    }
}