import React, {Component} from 'react'
import {Spin} from "antd";
import '../css/loadingSpin.css'

class LoadingSpin extends Component {
    render() {
        return (
            <div className='loadingSpin'>
                <Spin size={this.props.size} tip={this.props.tip}/>
            </div>
        )
    }
}

LoadingSpin.defaultProps = {
    size: 'large',
    tip: 'NGN服务器狂奔中……'
};
export default LoadingSpin;