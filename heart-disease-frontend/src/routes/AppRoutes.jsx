import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "../pages/Home";
import Test from "../pages/Test";
import Result from "../pages/Result";

function AppRoutes() {
    return (
    <BrowserRouter>
        <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/test" element={<Test />} />
        <Route path="/result" element={<Result />} />
        </Routes>
    </BrowserRouter>
    );
}

export default AppRoutes;