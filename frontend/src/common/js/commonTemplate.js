import React, {Component} from 'react';
import {BrowserRouter as Router, Route, Link} from "react-router-dom"
import Home from '../../home/js/home'
import '../css/commonTemplate.css'
import UserProfile from '../../userProfile/js/userProfile'
import MovieProfile from '../../movieProfile/js/movieProfile'

class CommonHeader extends Component {
    render() {
        return (
            <div id="header">
                <div className="inner">
                    <header>
                        <h1><a href="#" id="logo">NGN影评</a></h1>
                        <hr/>
                        <span className="byline">基于知识图谱的情感分析系统</span>
                    </header>
                    <footer>
                        <a href="#Content" className="button circled special scrolly">Go</a>
                    </footer>
                </div>

                <nav id="nav">
                    <ul>
                        <li><Link to="/">主页</Link></li>
                        <li><Link to="/knowledgeGraph">知识图谱</Link></li>
                        <li><Link to="/reviewAnalysis">评论解析</Link></li>
                        <li><Link to="/movieProfile">电影风评</Link></li>
                        <li><Link to="/userProfile">用户画像</Link></li>
                    </ul>
                </nav>

            </div>
        )
    }
}

class CommonFooter extends Component {
    render() {
        return (
            <div id="footer">
                <div className="container">
                    <div className="row">
                        <div className="12u">

                            {/*<!-- Contact -->*/}
                            <section className="contact">
                                <header>
                                    <h3>想要了解更多？</h3>
                                </header>
                                <p>请移步 <a href="http://203.91.121.76/">清华大学NGN实验室</a></p>
                                <ul className="icons">
                                    <li><a href="#" className="icon icon-twitter"><span>Twitter</span></a></li>
                                    <li><a href="#" className="icon icon-facebook"><span>Facebook</span></a></li>
                                    <li><a href="#" className="icon icon-google-plus"><span>Google+</span></a></li>
                                    <li><a href="#" className="icon icon-pinterest"><span>Pinterest</span></a></li>
                                    <li><a href="#" className="icon icon-dribbble"><span>Dribbble</span></a></li>
                                    <li><a href="#" className="icon icon-linkedin"><span>Linkedin</span></a></li>
                                </ul>
                            </section>

                            {/*<!-- Copyright -->*/}
                            <div className="copyright">
                                <ul className="menu">
                                    <li>&copy; <a href="https://www.tsinghua.edu.cn">Tsinghua University</a>. All rights
                                        reserved.
                                    </li>
                                    <li><a href="http://www.ee.tsinghua.edu.cn/">Department of Electronic
                                        Engineering</a></li>
                                    <li><a href="http://203.91.121.76/">Cooperation Contact</a></li>
                                </ul>
                            </div>

                        </div>

                    </div>
                </div>
            </div>
        )
    }
}


class CommonTemplate extends Component {
    render() {
        return (
            <Router>
                <div>
                    <CommonHeader/>
                    <Route exact path="/" component={Home}/>
                    <Route path="/userProfile" component={UserProfile}/>
                    <Route path="/movieProfile/:movieID" component={GetMovieProfile}/>
                    <CommonFooter/>
                </div>
            </Router>
        );
    }
}

function GetMovieProfile({match}) {
    return (
        <MovieProfile movieID={match.params.movieID}/>
    );
}

export default CommonTemplate