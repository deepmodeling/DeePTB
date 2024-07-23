
! calculate full ac conductivity using Kubo-Greenwood formula
! this ac_cond_f is originaly from tbplas code.
subroutine ac_cond_f(delta_eng, num_orb, num_kpt, prod_df, &
    omegas, num_omega, delta, &
    k_min, k_max, ac_cond)
implicit none

! input and output
real(kind=8), intent(in) :: delta_eng(num_orb, num_orb, num_kpt)
integer, intent(in) :: num_orb, num_kpt
complex(kind=8), intent(in) :: prod_df(num_orb, num_orb, num_kpt)
real(kind=8), intent(in) :: omegas(num_omega)
integer, intent(in) :: num_omega
real(kind=8), intent(in) :: delta
integer, intent(in) :: k_min, k_max
complex(kind=8), intent(inout) :: ac_cond(num_omega)

! local variables
integer :: i_w, i_k, mm, nn
real(kind=8) :: omega
complex(kind=8) :: cdelta, ac_sum

! calculate ac_cond
cdelta = dcmplx(0.0D0, delta)
!$OMP PARALLEL DO PRIVATE(omega, ac_sum, i_k, mm, nn)
do i_w = 1, num_omega
omega = omegas(i_w)
ac_sum = dcmplx(0.0D0, 0.0D0)
do i_k = k_min, k_max
do mm = 1, num_orb
do nn = 1, num_orb
   ac_sum = ac_sum + prod_df(nn, mm, i_k) &
          / (delta_eng(nn, mm, i_k) - omega - cdelta)
end do
end do
end do
ac_cond(i_w) = ac_sum
end do
!$OMP END PARALLEL DO
end subroutine ac_cond_f


subroutine ac_cond_gauss(delta_eng, num_orb, num_kpt, prod_df, &
    omegas, num_omega, delta, &
    k_min, k_max, ac_cond)
implicit none

! input and output
real(kind=8), intent(in) :: delta_eng(num_orb, num_orb, num_kpt)
integer, intent(in) :: num_orb, num_kpt
complex(kind=8), intent(in) :: prod_df(num_orb, num_orb, num_kpt)
real(kind=8), intent(in) :: omegas(num_omega)
integer, intent(in) :: num_omega
real(kind=8), intent(in) :: delta
integer, intent(in) :: k_min, k_max
complex(kind=8), intent(inout) :: ac_cond(num_omega)

real(kind=8), parameter :: PI = 3.141592653589793d0

! local variables
integer :: i_w, i_k, mm, nn
real(kind=8) :: omega
complex(kind=8) :: cdelta, ac_sum

! calculate ac_cond
cdelta = dcmplx(0.0D0, delta)
!$OMP PARALLEL DO PRIVATE(omega, ac_sum, i_k, mm, nn)
do i_w = 1, num_omega
omega = omegas(i_w)
ac_sum = dcmplx(0.0D0, 0.0D0)
do i_k = k_min, k_max
do mm = 1, num_orb
do nn = 1, num_orb
    ac_sum = ac_sum + prod_df(nn, mm, i_k) &
          *  exp(-0.5 * (( delta_eng(nn, mm, i_k) - omega) / delta)**2) & 
          / (delta * sqrt(2 * PI))
end do
end do
end do
ac_cond(i_w) = ac_sum
end do
!$OMP END PARALLEL DO
end subroutine ac_cond_gauss