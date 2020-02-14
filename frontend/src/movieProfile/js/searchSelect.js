import React, {Component} from 'react';
import {Select, Spin} from 'antd';
import debounce from 'lodash/debounce';


export class SearchSelect extends Component {

    constructor(props) {
        super(props);
        this.fetchData = debounce(this.fetchData, 800);
    }

    state = {
        data: [],
        fetching: false,
        value: undefined
    };
    fetchData = value => {
        this.setState({data: [], fetching: true});
        this.props.queryFunction(value, (data) => {
            this.setState({data, fetching: false});
        });
    };

    handleChange = value => {
        this.setState({
            data: [],
            fetching: false,
            value
        }, () => this.props.setValue(value));
    };

    render() {
        const {fetching, data, value} = this.state;
        console.log(this.state);
        // const {value} = this.props;
        return (
            <Select
                // labelInValue
                showSearch
                value={value}
                placeholder="搜索你关注的属性"
                notFoundContent={fetching ? <Spin size="small"/> : null}
                filterOption={false}
                onSearch={this.fetchData}
                onChange={this.handleChange}
                style={{width: '60%'}}
            >
                {/*<Select.Option key={1} value={1}>1</Select.Option>*/}
                {/*<Select.Option key={2} value={2}>2</Select.Option>*/}
                {this.props.makeOptions(data)}
            </Select>
        );
    }
}
