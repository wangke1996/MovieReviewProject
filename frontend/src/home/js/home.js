import React, {Component} from 'react';
import '../css/home.css';
import MovieInTheater from '../../home/js/movieInTheater'

class Home extends Component {

    render() {
        let prefix = process.env.PUBLIC_URL;
        return (
            <div id="Content" className="Home">
                <div id="banner">
                    <h2>Hi! 欢迎来到 <strong>NGN影评</strong>.</h2>
                    <span className="byline">
                        这是一个简单的演示系统，利用<a href="/knowledgeGraph">知识图谱</a>
                        对最新电影评论进行<a href="/reviewAnalysis">情感分析</a>，
                        从而挖掘用户观点或电影风评。
                    </span>
                    <hr/>
                </div>
                <MovieInTheater/>

                <div className="wrapper">
                    <section
                        id="features"
                        className="container special">
                        <header>
                            <h2> 系统功能 </h2>
                            <span className="byline">Ipsum volutpat consectetur orci metus consequat imperdiet duis integer semper magna.</span>
                        </header>
                        <div className="row">
                            <article className="3u special">
                                <a href="/knowledgeGraph" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic07.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="/knowledgeGraph">知识图谱</a></h3>
                                </header>
                                <p>
                                    电影领域知识图谱可视化展示，包括常见的评价对象、描述词和它们的相关关联
                                </p>
                            </article>
                            <article className="3u special">
                                <a href="/reviewAnalysis" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic08.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="/reviewAnalysis">评论解析</a></h3>
                                </header>
                                <p>
                                    上传单条或批量影评获取电影的细粒度评价
                                </p>
                            </article>
                            <article className="3u special">
                                <a href="/movieProfile" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic09.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="/movieProfile">电影风评</a></h3>
                                </header>
                                <p>
                                    查询当下或历史热门电影，获取该电影的细粒度评价
                                </p>
                            </article>
                            <article className="3u special">
                                <a href="/userProfile" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic10.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="/userProfile">用户画像</a></h3>
                                </header>
                                <p>
                                    上传用户的评论，获取用户的电影审美
                                </p>
                            </article>
                        </div>
                    </section>
                    <hr/>
                </div>

                <div className="wrapper style2">
                    <article id="contact" className="container special">
                        <header>
                            <h2>联系我们</h2>
                            <span className="byline">您在使用本系统过程中有什么疑问？<br/> 欢迎提出任何意见或建议</span>
                        </header>
                        <form action="#" method="post">
                            <div className="row">
                                <div className="6u">
                                    <input name="name" id="name" type="text" placeholder="姓名"/>
                                </div>
                                <div className="6u">
                                    <input name="email" id="email" type="email" placeholder="电子邮件"/>
                                </div>
                            </div>
                            <div className="row">
                                <div className="12u">
                                    <input name="subject" id="subject" type="text" placeholder="主题"/>
                                </div>
                            </div>
                            <div className="row">
                                <div className="12u">
                                                    <textarea name="message" id="message" placeholder="请详细描述您遇到的问题"
                                                              rows="6"/>
                                </div>
                            </div>
                            <div className="row">
                                <div className="12u">
                                    <ul className="align-center">
                                        <li><input type="submit" className="button special"
                                                   value="提交反馈"/></li>
                                    </ul>
                                </div>
                            </div>
                        </form>
                    </article>
                </div>

            </div>
        )
    }
}


export default Home;
