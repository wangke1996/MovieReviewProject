import React, {Component} from 'react'
import {Card, Divider, Popover, Rate} from "antd";
import {Axis, Chart, Coord, Geom, Label, Tooltip as BizTooltip} from "bizcharts";
import {image_url} from "../../libs/toolFunctions";

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

class BasicInfo extends Component {
    render() {
        const data = this.props.data;

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

        return (
            <section>
                <header>
                    <h2>总体评分<span
                        className="emphatic">{data['rating']['average']}</span>/{data['rating']['max']}
                    </h2>
                    <span className="byline">
                                    <Rate disabled allowHalf defaultValue={parseInt(data['rating']['stars']) / 10}/>
                        {data['ratings_count']}人参与评价
                                </span>
                </header>
                <Chart height={250} data={get_rate_data()} forceFit>
                    <Coord transpose/>
                    <Axis name="rate"/>
                    <Axis name="num"/>
                    <BizTooltip/>
                    <Geom type="interval" position="rate*num">
                        <Label content='num' offset={0} textStyle={
                            {
                                textAlign: 'right',
                                fill: '#ffae6b'
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
                    <p>上映日期：{data['mainland_pubdate']}</p>
                    <p>片长：{data['durations']}</p>
                    <p>别名：{data['aka'].join('/')}</p>
                </div>
            </section>
        );
    }
}

export default BasicInfo