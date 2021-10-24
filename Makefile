PYTHON?=python
VERSION?=$(shell sed -ne "s|^__version__\s*=\s*'\([^']*\)'.*|\1|p" src/quicktions.pyx)
PACKAGE=quicktions
WITH_CYTHON := $(shell python -c 'from Cython.Build import cythonize' 2>/dev/null && echo "--with-cython")

MANYLINUX_IMAGES= \
	manylinux2010_x86_64 \
	manylinux2010_i686 \
	manylinux2014_aarch64

.PHONY: all local sdist test clean realclean wheel_manylinux

all:  local

local:
	${PYTHON} setup.py build_ext --inplace $(WITH_CYTHON)

sdist: dist/$(PACKAGE)-$(VERSION).tar.gz

dist/$(PACKAGE)-$(VERSION).tar.gz:
	$(PYTHON) setup.py sdist $(WITH_CYTHON)

testslow: local
	PYTHONPATH=src $(PYTHON) src/test_fractions.py

test: local
	PYTHONPATH=src $(PYTHON) src/test_fractions.py --fast

clean:
	rm -fr build src/*.so

realclean: clean
	rm -fr src/*.c src/*.html

qemu-user-static:
	docker run --rm --privileged hypriot/qemu-register

wheel_manylinux: sdist $(addprefix wheel_,$(MANYLINUX_IMAGES))
$(addprefix wheel_,$(filter-out %_x86_64, $(filter-out %_i686, $(MANYLINUX_IMAGES)))): qemu-user-static

wheel_%: dist/$(PACKAGE)-$(VERSION).tar.gz
	echo "Building wheels for $(PACKAGE) $(VERSION)"
	mkdir -p wheelhouse_$(subst wheel_,,$@)
	time docker run --rm -t \
		-v $(shell pwd):/io \
		-e CFLAGS="-O3 -g0 -mtune=generic -pipe -fPIC" \
		-e LDFLAGS="$(LDFLAGS) -fPIC" \
		-e WHEELHOUSE=wheelhouse$(subst wheel_manylinux,,$@) \
		quay.io/pypa/$(subst wheel_,,$@) \
		bash -c 'for PYBIN in /opt/python/cp*/bin; do \
		    $$PYBIN/python -V; \
		    { $$PYBIN/pip wheel -w /io/$$WHEELHOUSE /io/$< & } ; \
		    done; wait; \
		    for whl in /io/$$WHEELHOUSE/$(PACKAGE)-$(VERSION)-*-linux_*.whl; do auditwheel repair $$whl -w /io/$$WHEELHOUSE; done'
