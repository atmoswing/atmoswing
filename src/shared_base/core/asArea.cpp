/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asArea.h"
#include "asIncludes.h"

asArea::asArea(const Coo& cornerUL, const Coo& cornerUR, const Coo& cornerLL, const Coo& cornerLR, int flatAllowed,
               bool isLatLon)
    : _cornerUL(cornerUL),
      _cornerUR(cornerUR),
      _cornerLL(cornerLL),
      _cornerLR(cornerLR),
      _flatAllowed(flatAllowed),
      _isLatLon(isLatLon) {
    Init();
}

asArea::asArea(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed, bool isLatLon)
    : _flatAllowed(flatAllowed),
      _isLatLon(isLatLon) {
    if (flatAllowed == asFLAT_ALLOWED) {
        yWidth = std::max(yWidth, 0.0);
        xWidth = std::max(xWidth, 0.0);
    }

    _cornerUL = {xMin, yMin + yWidth};
    _cornerUR = {xMin + xWidth, yMin + yWidth};
    _cornerLL = {xMin, yMin};
    _cornerLR = {xMin + xWidth, yMin};

    Init();
}

asArea::asArea()
    : _cornerUL({0, 0}),
      _cornerUR({0, 0}),
      _cornerLL({0, 0}),
      _cornerLR({0, 0}),
      _flatAllowed(asFLAT_ALLOWED),
      _isLatLon(true) {}

void asArea::Init() {
    if (_isLatLon) DoCheckPoints();
    if (!CheckConsistency()) throw std::runtime_error(_("Unable to build a consistent area with the given coordinates."));
    if (!IsRectangle()) throw std::runtime_error(_("The provided area is not rectangle."));
}

void asArea::DoCheckPoints() {
    CheckPoint(_cornerUL);
    CheckPoint(_cornerUR);
    CheckPoint(_cornerLL);
    CheckPoint(_cornerLR);
}

void asArea::CheckPoint(Coo& point) {
    if (point.y < -90) {
        point.y = -90;
    }
    if (point.y > 90) {
        point.y = 90;
    }
}

bool asArea::CheckConsistency() {
    Coo cootmp;

    if (_flatAllowed == asFLAT_FORBIDDEN) {
        if ((_cornerUL.x == _cornerUR.x) || (_cornerLL.x == _cornerLR.x) || (_cornerLL.y == _cornerUL.y) ||
            (_cornerLR.y == _cornerUR.y)) {
            return false;
        }
    }

    if (_cornerUL.x > _cornerUR.x) {
        cootmp = _cornerUR;
        _cornerUR = _cornerUL;
        _cornerUL = cootmp;
    }

    if (_cornerLL.x > _cornerLR.x) {
        cootmp = _cornerLR;
        _cornerLR = _cornerLL;
        _cornerLL = cootmp;
    }

    if (_cornerLL.y > _cornerUL.y) {
        cootmp = _cornerUL;
        _cornerUL = _cornerLL;
        _cornerLL = cootmp;
    }

    if (_cornerLR.y > _cornerUR.y) {
        cootmp = _cornerUR;
        _cornerUR = _cornerLR;
        _cornerLR = cootmp;
    }

    return true;
}

double asArea::GetXmin() const {
    return std::min(_cornerUL.x, _cornerLL.x);
}

double asArea::GetXmax() const {
    return std::max(_cornerUR.x, _cornerLR.x);
}

double asArea::GetXwidth() const {
    return std::abs(_cornerUR.x - _cornerUL.x);
}

double asArea::GetYmin() const {
    return std::min(std::min(_cornerUL.y, _cornerLL.y), std::min(_cornerUR.y, _cornerLR.y));
}

double asArea::GetYmax() const {
    return std::max(std::max(_cornerUL.y, _cornerLL.y), std::max(_cornerUR.y, _cornerLR.y));
}

double asArea::GetYwidth() const {
    return std::abs(_cornerUR.y - _cornerLR.y);
}

bool asArea::IsRectangle() const {
    // Check that the area is a square
    return !((_cornerUL.x != _cornerLL.x) | (_cornerUL.y != _cornerUR.y) | (_cornerUR.x != _cornerLR.x) |
             (_cornerLL.y != _cornerLR.y));
}
