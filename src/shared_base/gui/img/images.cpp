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
 * Portions Copyright 2015 Pascal Horton, Terranum.
 */

#ifndef WX_PRECOMP

#endif

#include "images.h"

void initialize_images(double ppiScaling) {
    if (ppiScaling <= 1.25) {
        initialize_img_misc_s();
        initialize_img_toolbar_s();
        initialize_img_treectrl_s();
    } else if (ppiScaling <= 1.75) {
        initialize_img_misc_m();
        initialize_img_toolbar_m();
        initialize_img_treectrl_m();
    } else {
        initialize_img_misc_l();
        initialize_img_toolbar_l();
        initialize_img_treectrl_l();
    }

    initialize_img_logo();
}

void cleanup_images() {
    cleanup_img_misc();
    cleanup_img_toolbar();
    cleanup_img_treectrl();
    cleanup_img_logo();
}