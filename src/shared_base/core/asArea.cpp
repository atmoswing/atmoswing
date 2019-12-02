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

asArea::asArea(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR, int flatAllowed,
               bool isLatLon)
    : m_cornerUL(cornerUL),
      m_cornerUR(cornerUR),
      m_cornerLL(cornerLL),
      m_cornerLR(cornerLR),
      m_flatAllowed(flatAllowed),
      m_isLatLon(isLatLon) {
  Init();
}

asArea::asArea(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed, bool isLatLon)
    : m_flatAllowed(flatAllowed),
      m_isLatLon(isLatLon) {
  if (flatAllowed == asFLAT_ALLOWED) {
    yWidth = wxMax(yWidth, 0.0);
    xWidth = wxMax(xWidth, 0.0);
  } else {
    wxASSERT(yWidth > 0);
    wxASSERT(xWidth > 0);
  }

  m_cornerUL = {xMin, yMin + yWidth};
  m_cornerUR = {xMin + xWidth, yMin + yWidth};
  m_cornerLL = {xMin, yMin};
  m_cornerLR = {xMin + xWidth, yMin};

  Init();
}

asArea::asArea()
    : m_cornerUL({0, 0}),
      m_cornerUR({0, 0}),
      m_cornerLL({0, 0}),
      m_cornerLR({0, 0}),
      m_flatAllowed(asFLAT_ALLOWED),
      m_isLatLon(true) {}

void asArea::Init() {
  if (m_isLatLon && !DoCheckPoints()) asThrowException(_("Use asAreaComp in this case."));
  if (!CheckConsistency()) asThrowException(_("Unable to build a consistent area with the given coordinates."));
  if (!IsRectangle()) asThrowException(_("The provided area is not rectangle."));
}

bool asArea::DoCheckPoints() {
  return !(!CheckPoint(m_cornerUL, asEDIT_FORBIDDEN) || !CheckPoint(m_cornerUR, asEDIT_FORBIDDEN) ||
           !CheckPoint(m_cornerLL, asEDIT_FORBIDDEN) || !CheckPoint(m_cornerLR, asEDIT_FORBIDDEN));
}

bool asArea::CheckPoint(Coo &point, int changesAllowed) {
  // We always consider WGS84 for the predictors
  if (point.y < -90) {
    return false;
  }
  if (point.y > 90) {
    return false;
  }
  if (point.x < 0) {
    if (changesAllowed == asEDIT_ALLOWED) {
      point.x += 360;
    }
    return false;
  }
  if (point.x > 360) {
    if (changesAllowed == asEDIT_ALLOWED) {
      point.x -= 360;
    }
    return false;
  }

  return true;
}

bool asArea::CheckConsistency() {
  Coo cootmp;

  if (m_flatAllowed == asFLAT_FORBIDDEN) {
    if ((m_cornerUL.x == m_cornerUR.x) || (m_cornerLL.x == m_cornerLR.x) || (m_cornerLL.y == m_cornerUL.y) ||
        (m_cornerLR.y == m_cornerUR.y)) {
      return false;
    }
  }

  if (m_cornerUL.x > m_cornerUR.x) {
    cootmp = m_cornerUR;
    m_cornerUR = m_cornerUL;
    m_cornerUL = cootmp;
  }

  if (m_cornerLL.x > m_cornerLR.x) {
    cootmp = m_cornerLR;
    m_cornerLR = m_cornerLL;
    m_cornerLL = cootmp;
  }

  if (m_cornerLL.y > m_cornerUL.y) {
    cootmp = m_cornerUL;
    m_cornerUL = m_cornerLL;
    m_cornerLL = cootmp;
  }

  if (m_cornerLR.y > m_cornerUR.y) {
    cootmp = m_cornerUR;
    m_cornerUR = m_cornerLR;
    m_cornerLR = cootmp;
  }

  return true;
}

double asArea::GetXmin() const {
  return wxMin(m_cornerUL.x, m_cornerLL.x);
}

double asArea::GetXmax() const {
  return wxMax(m_cornerUR.x, m_cornerLR.x);
}

double asArea::GetXwidth() const {
  return std::abs(m_cornerUR.x - m_cornerUL.x);
}

double asArea::GetYmin() const {
  return wxMin(wxMin(m_cornerUL.y, m_cornerLL.y), wxMin(m_cornerUR.y, m_cornerLR.y));
}

double asArea::GetYmax() const {
  return wxMax(wxMax(m_cornerUL.y, m_cornerLL.y), wxMax(m_cornerUR.y, m_cornerLR.y));
}

double asArea::GetYwidth() const {
  return std::abs(m_cornerUR.y - m_cornerLR.y);
}

bool asArea::IsRectangle() const {
  // Check that the area is a square
  return !((m_cornerUL.x != m_cornerLL.x) | (m_cornerUL.y != m_cornerUR.y) | (m_cornerUR.x != m_cornerLR.x) |
           (m_cornerLL.y != m_cornerLR.y));
}
